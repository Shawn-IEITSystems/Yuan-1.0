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

"""Evaluation utilities."""

import os
import time
from functools import partial

import torch

from megatron import get_args, get_timers
from megatron import print_rank_last, is_last_rank
from megatron import mpu
from megatron.schedules import get_forward_backward_func
from tasks.finetune_utils import build_data_loader, build_eval_data_loader
from tasks.finetune_utils import process_batch
import json
import re
def accuracy_func_provider(single_dataset_provider):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    datapaths = args.valid_data
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_data_loader(
            dataset, args.orig_micro_batch_size, num_workers=args.num_workers,
            drop_last=(mpu.get_data_parallel_world_size() > 1))
        dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_last('calculating metrics ...')
        correct = 0
        total = 0
        if output_predictions:
            assert mpu.get_data_parallel_world_size() == 1
            named_predictions = []
            names = 'predictions'
        for name, dataloader in dataloaders:
            output = calculate_correct_answers(name, model, dataloader,
                                               epoch, output_predictions)
            if not output_predictions:
                correct_ans, total_count = output
            else:
                correct_ans, total_count, predictions = output
                named_predictions.append((name, predictions))
                names += '_' + name
            correct += correct_ans
            total += total_count
        if is_last_rank():
            percent = float(correct) * 100.0 / float(total)
            print(' >> |epoch: {}| overall: correct / total = {} / {} = '
                  '{:.4f} %'.format(epoch, correct, total, percent))

        if output_predictions and is_last_rank():
            assert args.load is not None
            filename = os.path.join(args.load, names + '.pt')
            torch.save(named_predictions, filename)

    return metrics_func


def calculate_correct_answers(name, model, dataloader,
                              epoch, output_predictions):
    """Calculate correct over total answers and return prediction if the
    `output_predictions` is true."""
    args = get_args()
    timers = get_timers()
    forward_backward_func = get_forward_backward_func()
    start_time = time.time()
    for m in model:
        m.eval()
    saved_micro_batch_size = args.micro_batch_size
    saved_global_batch_size = args.global_batch_size
    ds = dataloader.dataset
    if hasattr(ds, 'sample_multiplier'):
        # If our dataset as a sample_multiplier attribute that means
        # each "sample" from the dataset actually has multiple samples
        # that will collapse into the batch dimension (for example in
        # the RACE dataset that has several options), we need to
        # account for that when setting the micro batch size.
        sample_multiplier = ds.sample_multiplier
    else:
        sample_multiplier = 1
    micro_batch_size_times_data_parallel = args.orig_micro_batch_size * args.data_parallel_size
    num_micro_batches = args.orig_global_batch_size // micro_batch_size_times_data_parallel

    def loss_func(output_predictions, labels, output_tensor):
        logits = output_tensor

        loss_dict = {}
        # Add output predictions.
        if output_predictions:
            assert False
            loss_dict['softmaxes'] = torch.nn.Softmax(dim=-1)(
                logits.float()).data.cpu().numpy().tolist()
            loss_dict['labels'] = labels.data.cpu().numpy().tolist()
            loss_dict['ids'] = batch['uid'].cpu().numpy().tolist()
        # Compute the correct answers.
        predicted = torch.argmax(logits, dim=-1)
        corrects = (predicted == labels)
        # Add to the counters.
        loss_dict['total'] = labels.size(0)
        loss_dict['correct'] = corrects.sum().item()

        return 0, loss_dict

    # defined inside to capture output_predictions
    def correct_answers_forward_step(batch, model, unwrapped_model=None, input_tensor=None):
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        # tokens, types, labels, attention_mask = process_batch(batch_)
        timers('batch-generator').start()
        tokens, labels, _, attention_mask, position_ids, pooling_sequence_indexs = process_batch(batch_)
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
                output_tensor = model(tokens[:,s_token:e_token,:], position_ids[:,s_token:e_token,:], attention_mask[:,s_token:e_token,:,:], pooling_sequence_indexs=pooling_sequence_indexs[:,s_token:e_token])
                if s_token == 0:
                    output_tensors =  output_tensor
                else:
                    output_tensors = torch.cat([output_tensors, output_tensor], 1)
                s_token += args.reset_batch
            output_tensor = output_tensors
        return output_tensor, partial(loss_func, output_predictions, labels)

    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        correct = 0
        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1
            softmaxes = []
            labels = []
            ids = []

        for _, batch in enumerate(dataloader):
            actual_batch_size = len(batch['label'])
            # ... applying sample_multiplier if necessary
            args.micro_batch_size = actual_batch_size * sample_multiplier
            args.global_batch_size = actual_batch_size * sample_multiplier * num_micro_batches
            #print(batch['label'])

            loss_dicts = forward_backward_func(correct_answers_forward_step, batch, model,
                                               optimizer=None, timers=None, forward_only=True)
            for loss_dict in loss_dicts:
                if output_predictions:
                    softmaxes.extend(loss_dict['softmaxes'])
                    labels.extend(loss_dict['labels'])
                    ids.extend(loss_dict['ids'])
                total += loss_dict['total']
                correct += loss_dict['correct']

    for m in model:
        m.train()
    args.micro_batch_size = saved_micro_batch_size
    args.global_batch_size = saved_global_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage():
        unreduced = torch.cuda.LongTensor([correct, total])
        torch.distributed.all_reduce(unreduced,
                                     group=mpu.get_data_parallel_group())

        # Print on screen.
        correct_ans = unreduced[0].item()
        total_count = unreduced[1].item()
        percent = float(correct_ans) * 100.0 / float(total_count)
        elapsed_time = time.time() - start_time
        print_rank_last(' > |epoch: {}| metrics for {}: correct / total '
                        '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
                            epoch, name, correct_ans, total_count,
                            percent, elapsed_time))

        if output_predictions:
            return correct_ans, total_count, (softmaxes, labels, ids)
        return correct_ans, total_count
    if output_predictions:
        return 0, 0, ()
    return 0, 0

def predict_test_func_provider(single_dataset_provider):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    datapaths = args.test_data
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_eval_data_loader(dataset)
        dataloaders.append((dataset.dataset_name, dataloader))
    
        args.micro_batch_size *= dataset.sample_multiplier
    def predict_test_func(model, epoch, output_predictions=False):
        print_rank_last('generate test predict result ...')
        
        if output_predictions:
            assert mpu.get_data_parallel_world_size() == 1

        for name, dataloader in dataloaders:
            _ = generate_predict_answers(name, model, dataloader,
                                               epoch, output_predictions)
    return predict_test_func


def generate_predict_answers(name, model, dataloader,
                              epoch, output_predictions):
    """Calculate correct over total answers and return prediction if the
    `output_predictions` is true."""
    args = get_args()
    timers = get_timers()
    forward_backward_func = get_forward_backward_func()

    for m in model:
        m.eval()
    saved_micro_batch_size = args.micro_batch_size
    saved_global_batch_size = args.global_batch_size
    # ds = dataloader.dataset
    # if hasattr(ds, 'sample_multiplier'):
    #     # If our dataset as a sample_multiplier attribute that means
    #     # each "sample" from the dataset actually has multiple samples
    #     # that will collapse into the batch dimension (for example in
    #     # the RACE dataset that has several options), we need to
    #     # account for that when setting the micro batch size.
    #     sample_multiplier = ds.sample_multiplier
    # else:
    #     sample_multiplier = 1
    # micro_batch_size_times_data_parallel = args.orig_micro_batch_size * args.data_parallel_size
    # num_micro_batches = args.orig_global_batch_size // micro_batch_size_times_data_parallel

    labels_predicted = []
    # def predict_func(output_tensor):
    #     logits = output_tensor
    #     predicted = torch.argmax(logits, dim=-1)
    #     return predicted

    # defined inside to capture output_predictions
    def predict_answers_forward_step(batch, model, unwrapped_model=None, input_tensor=None):
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        timers('batch-generator').start()
        tokens, _, _, attention_mask, position_ids, pooling_sequence_indexs = process_batch(batch_)
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
                output_tensor = model(tokens[:,s_token:e_token,:], position_ids[:,s_token:e_token,:], attention_mask[:,s_token:e_token,:,:], pooling_sequence_indexs=pooling_sequence_indexs[:,s_token:e_token])
                if s_token == 0:
                    output_tensors =  output_tensor
                else:
                    output_tensors = torch.cat([output_tensors, output_tensor], 1)
                s_token += args.reset_batch
            output_tensor = output_tensors
        if mpu.is_pipeline_last_stage():
            output_tensor = torch.argmax(output_tensor, dim=-1)
        return output_tensor, None

    with torch.no_grad():
        # For all the batches in the dataset.

        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1

        labels_predicted = []
        for _, batch in enumerate(dataloader):
            # ... applying sample_multiplier if necessary
            if not mpu.is_pipeline_last_stage():
                forward_backward_func(predict_answers_forward_step, batch, model,
                                                   optimizer=None, timers=None, forward_only=True)
            else:
                label_predicted = forward_backward_func(predict_answers_forward_step, batch, model,
                                                   optimizer=None, timers=None, forward_only=True)
                labels_predicted.append(label_predicted[0].item())

    for m in model:
        m.train()
    args.micro_batch_size = saved_micro_batch_size
    args.global_batch_size = saved_global_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage() and mpu.get_data_parallel_rank()==0 and args.local_rank==0:
        if args.task in ['CSL']:
            save_csl_result(labels_predicted, epoch)
        elif args.task in ['BUSTM']:
            save_bustm_result(labels_predicted, epoch)
        elif args.task in ['WSC']:
            save_wsc_result(labels_predicted, epoch)
        elif args.task in ['OCNLI']:
            save_ocnli_result(labels_predicted, epoch)
        elif args.task in ['TNEWS']:
            save_tnews_result(labels_predicted, epoch)
        elif args.task in ['CSLDCP']:
            save_csldcp_result(labels_predicted, epoch)
        elif args.task in ['EPRSTMT']:
            save_eprstmt_result(labels_predicted, epoch)
        elif args.task in ['CHID']:
            save_chid_result(labels_predicted, epoch)
        elif args.task in ['IFLYTEK']:
            save_iflytek_result(labels_predicted, epoch)
        else:
            raise NotImplementedError('Task {} is not implemented.'.format(
                args.task))

    return 0, 0

def save_csl_result(labels_predicted, epoch):
    args = get_args()
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'cslf_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r = {}
                    r['id'] = d['id']
                    r['label'] = str(labels_predicted[i])
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')

def save_bustm_result(labels_predicted, epoch):
    args = get_args()
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'bustm_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r={}
                    r['id'] = d['id']
                    r['label'] = str(labels_predicted[i])
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')

def save_wsc_result(labels_predicted, epoch):
    args = get_args()
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'cluewscf_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, _ in enumerate(lines):

                    r = {}
                    if labels_predicted[i] == 0:
                        r['label'] = 'false'
                    else:
                        r['label'] = 'true'
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')

def save_ocnli_result(labels_predicted, epoch):
    args = get_args()
    labels = ["neutral", "entailment", "contradiction"]
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'ocnlif_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r = {}
                    r['id'] = d['id']
                    r['label'] = labels[labels_predicted[i]]
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')

def save_tnews_result(labels_predicted, epoch):
    args = get_args()
    with open(args.labels_path, 'r') as f:
        lines = f.readlines()
        LABELS_ID={}
        LABELS_ZH={}
        LABELS_EN={}
        LABELS_ID2={}
        for i, line in enumerate(lines):
            d = json.loads(line)
            LABELS_ID[i] = d['label']
            LABELS_EN[i] = d['label_desc']
            LABELS_ZH[i] = d['label_zh']
            LABELS_ID2[d['label']] = i
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'tnewsf_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r = {}
                    r['label'] = LABELS_ID[labels_predicted[i]]
                    r['id'] = d['id']
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')

def save_csldcp_result(labels_predicted, epoch):
    args = get_args()
    LABELS = {}
    with open(args.labels_path) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            LABELS[i] =l.replace('\n', '').replace('\/','\\\/')
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'csldcp_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            print_rank_last(os.path.join('./result_public',args.task, 'csldcp_predict'+'_'+str(epoch)+'.json'))
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r = {}
                    r['id'] = d['id']
                    r['label'] = LABELS[labels_predicted[i]]
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')



def save_eprstmt_result(labels_predicted, epoch):
    args = get_args()
    labels = ["Negative", "Positive"]
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'eprstmt_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r = {}
                    r['id'] = d['id']
                    r['label'] =  labels[labels_predicted[i]]
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')

                

def save_chid_result(labels_predicted, epoch):
    args = get_args()
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'chidf_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r = {}
                    r['id'] = d['id']
                    r['answer'] = labels_predicted[i]
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')

def save_iflytek_result(labels_predicted, epoch):
    args = get_args()
    LABELS = {}
    with open(args.labels_path) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            LABELS[i] =l.replace('\n', '').replace('\/','\\\/')
    for fname in args.test_data:
        with open(os.path.join('./result_public',args.task, 'iflytekf_predict'+'_'+str(epoch)+'.json'), 'w', encoding='utf-8') as fout:
            with open(fname, 'r',encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == len(labels_predicted), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(labels_predicted), len(lines), fname)
                for i, line in enumerate(lines):
                    d = json.loads(line)
                    r = {}
                    r['id'] = d['id']
                    r['label'] = str(labels_predicted[i])
                    s = json.dumps(r, ensure_ascii=False)
                    fout.write(s + '\n')
