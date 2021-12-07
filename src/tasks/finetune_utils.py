# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetune utilities."""

from functools import partial
import sys
import torch

from megatron import get_args, get_num_microbatches
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import evaluate_and_print_results
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import calc_params_l2_norm
from megatron.utils import check_adlr_autoresume_termination
from megatron import get_tokenizer
from megatron.schedules import forward_step
import numpy as np
from tqdm import tqdm
import os

def iid_noise_mask(length, noise_density):
  """Independent and identically distributed token noise.

  Args:
    length: an int32 scalar
    noise_density: a float - approximate density of output mask

  Returns:
    a boolean tensor with shape [length]
  """
  return torch.rand(length) < noise_density

def noise_to_sentinel(tokens, noise_mask, sentinel_id):
  """Replace each run of noise tokens with a single sentinel.

  Args:
    tokens: a 1d integer Tensor
    noise_mask: a boolean Tensor with the same shape as tokens
    sentinel_id: a mask id in vocabulary
  Returns:
    a Tensor with the same shape and dtype as tokens
  """
  tokens = torch.where(noise_mask.to(tokens.device), sentinel_id, tokens)
  return tokens

def get_prefix_lm_masks_and_position_ids(data,
                                         padding_mask,
                                         eod_token,
                                         mask_token,
                                         reset_position_ids,
                                         reset_attention_mask,
                                         prefix_lm_mask,
                                         masked_lm_prob):
    """Build masks and position id for left to right model."""
    args = get_args()

    # Extract batch size and sequence length.
    micro_batch_size, num_choices, seq_length = data.size()

    # Attention mask (lower triangular).
    att_mask_batch = micro_batch_size

    attention_mask = torch.ones((att_mask_batch, num_choices, seq_length, seq_length), device=data.device)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()
    pooling_sequence_indexs = []
    # Loop through the batches:
    for b in range(micro_batch_size):
        # Randomly select prefix length
        pooling_sequence_index = []
        # Set mask for prefix
        for ci in range(num_choices):
            padding_index = position_ids[b, ci, padding_mask[b][ci] == 0]
          
            
            if len(padding_index) == 0:
                prefix_length = args.seq_length
            else:
                prefix_length = padding_index[0]
            pooling_sequence_index.append(prefix_length-2)
            attention_mask[b, ci, padding_index[0]-1:, :]=0
            attention_mask[b, ci, :, padding_index[0]-1:]=0
            
        pooling_sequence_indexs.append(pooling_sequence_index)
    pooling_sequence_indexs = torch.tensor(pooling_sequence_indexs, dtype=torch.long, device=data.device)

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)
    return data, attention_mask, loss_mask, position_ids, pooling_sequence_indexs

def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    tokens = batch['text'].long().cuda().contiguous()
    padding_mask = batch['padding_mask'].long().cuda().contiguous()
    labels = batch['label'].long().cuda().contiguous()

    # Get the masks and postition ids.
    tokens, attention_mask, loss_mask, position_ids, pooling_sequence_indexs = get_prefix_lm_masks_and_position_ids(
        tokens,
        padding_mask,
        tokenizer.eod,
        tokenizer.mask,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.prefix_lm_mask,
        args.masked_prefix_lm_prob)
    return tokens, labels, loss_mask, attention_mask, position_ids, pooling_sequence_indexs


def cross_entropy_loss_func(labels, loss_mask, output_tensor):
    logits = output_tensor
    loss_mask = loss_mask.view(-1).float()
    # Cross-entropy loss.
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(logits.contiguous().float(), labels)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def _cross_entropy_forward_step(batch, model, unwrapped_model=None, input_tensor=None):
    """Simple forward step with cross-entropy loss."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    tokens, labels, loss_mask, attention_mask, position_ids, pooling_sequence_indexs = process_batch(batch_)
    timers('batch-generator').stop()

    # Forward model.
    args = get_args()
    if args.reset_batch is None or args.task is None:
        output_tensor = model(tokens, position_ids, attention_mask, pooling_sequence_indexs=pooling_sequence_indexs)
    else:
        s_token = 0
        while s_token < tokens.shape[1]:
            e_token = min(tokens.shape[1],s_token+args.reset_batch)
            if input_tensor is not None:
                unwrapped_model.set_input_tensor(input_tensor[:,s_token:e_token])
            output_tensor = model(tokens[:,s_token:e_token,:], 
                                position_ids[:,s_token:e_token,:], 
                                attention_mask[:,s_token:e_token,:,:], 
                                pooling_sequence_indexs=pooling_sequence_indexs[:,s_token:e_token])
            if s_token == 0:
                output_tensors =  output_tensor
            else:
                output_tensors = torch.cat([output_tensors, output_tensor], 1)
            s_token += args.reset_batch
        output_tensor = output_tensors
    return output_tensor, partial(cross_entropy_loss_func, labels, loss_mask)


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=micro_batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=task_collate_fn)

    return data_loader

def build_eval_data_loader(dataset, task_collate_fn=None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    # Sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    args = get_args()
    # Data loader. Note that batch size is the per GPU batch size.
    num_workers = args.num_workers
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=False,
                                              pin_memory=True,
                                              collate_fn=task_collate_fn)

    return data_loader

def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset, 
    task_collate_fn=None):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset, args.micro_batch_size,
                                         args.num_workers, not args.keep_last,
                                         task_collate_fn)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader_ = build_data_loader(valid_dataset, args.micro_batch_size,
                                          args.num_workers, not args.keep_last,
                                          task_collate_fn)
    valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

    # Now that we've built the data loaders, set batch_size arguments
    # to the actual batch size the model will see for this dataset.
    # This is necessary so pipeline transfers know what size they are
    # and the LR schedule, which is based on samples seen, gets set
    # correctly.
    args.orig_micro_batch_size = args.micro_batch_size
    args.orig_global_batch_size = args.global_batch_size
    if hasattr(train_dataset, 'sample_multiplier'):
        # If our dataset as a sample_multiplier attribute that means
        # each "sample" from the dataset actually has multiple samples
        # that will collapse into the batch dimension (for example in
        # the RACE dataset that has several options), we need to
        # account for that when setting the micro batch size.
        args.micro_batch_size *= train_dataset.sample_multiplier
        args.global_batch_size *= train_dataset.sample_multiplier
        args.train_sample_multiplier = train_dataset.sample_multiplier
    if hasattr(valid_dataset, 'sample_multiplier'):
        args.valid_sample_multiplier = valid_dataset.sample_multiplier

    return train_dataloader, valid_dataloader

def _train(model, optimizer, lr_scheduler, forward_step,
           train_dataloader, valid_dataloader, end_of_epoch_callback, predict_callback):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    assert get_num_microbatches() == 1, "finetuning with gradient accumulation doesn't currently work"

    # Turn on training mode which enables dropout.
    for m in model:
        m.train()

    # Tracking loss.
    losses_dict_sum = {}

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration

    # Memory reporting flag.
    report_memory_flag = True
    # For each remaining epoch
    timers('interval-time').start()
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(args.seed + epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader):
            args.micro_batch_size = args.orig_micro_batch_size * args.train_sample_multiplier

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Train for one step.
            out = train_step(forward_step, batch, model, optimizer, lr_scheduler)

            losses_dict, skipped_iter, grad_norm, num_zeros_in_grad = out
            iteration += 1

            # Logging.
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)
            report_memory_flag = training_log(losses_dict, losses_dict_sum,
                                              optimizer.param_groups[0]['lr'],
                                              iteration,
                                              optimizer.get_loss_scale().item(),
                                              report_memory_flag, skipped_iter,
                                              grad_norm, params_norm, num_zeros_in_grad)

            # Autoresume
            if args.adlr_autoresume and \
               (iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(iteration, model,
                                                  optimizer, lr_scheduler)

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0:
                args.micro_batch_size = args.orig_micro_batch_size * args.valid_sample_multiplier
                prefix = 'iteration {}'.format(iteration)
                evaluate_and_print_results(prefix, forward_step,
                                           valid_dataloader, model,
                                           iteration, False)

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                save_checkpoint(iteration, model, optimizer, lr_scheduler)
                torch.distributed.barrier()
                print_rank_0('exiting program at iteration {}'.format(iteration))
                sys.exit()

            # Callback at the end of each epoch.
            if args.eval_interval and iteration % args.eval_interval == 0:
                if end_of_epoch_callback is not None:
                    args.micro_batch_size = args.orig_micro_batch_size * args.valid_sample_multiplier
                    end_of_epoch_callback(model, epoch)
        if args.save is not None:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)
        torch.distributed.barrier()
    


def finetune(train_valid_datasets_provider, model_provider,
             forward_step=_cross_entropy_forward_step,
             end_of_epoch_callback_provider=None,
             predict_callback_provider=None,
             task_collate_fn=None):
    """Main finetune function used across all tasks."""
    args = get_args()
    args.orig_micro_batch_size = args.micro_batch_size
    args.orig_global_batch_size = args.global_batch_size
    timers = get_timers()

    assert args.rampup_batch_size is None, \
        'batch size scaling is not supported for finetuning'

    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder').start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset, task_collate_fn)
    else:
        args.train_iters = 0
    timers('train/valid/test dataset/dataloder').stop()
    # Build calback function.
    timers('callback function').start()
    end_of_epoch_callback = None
    predict_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider()
    if predict_callback_provider is not None:
        predict_callback = predict_callback_provider()
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    if args.epochs > 0:
        model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, load_lr_scheduler=True)
    else:
        model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, load_lr_scheduler=False)
    timers('model and optimizer').stop()
    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint').start()
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        original_rng = args.no_load_rng
        args.no_load_rng = True
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        args.no_load_rng = original_rng
        # This is critical when only model is loaded. We should make sure
        # main parameters are also updated.
        if args.epochs > 0:
            optimizer.reload_model_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    if args.epochs > 0:
        _train(model, optimizer, lr_scheduler, forward_step,
               train_dataloader, valid_dataloader, end_of_epoch_callback, predict_callback)
    # Or just evaluate.
    else:
        if predict_callback is not None:
            print_rank_0('predict test result')
            predict_callback(model, epoch=args.eval_epochs, output_predictions=True)
    print_rank_0('done :-)')

