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

"""Utilities for generating text."""

import copy
import json
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
from megatron import get_args, print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
from tools.generate_loss_gpt_utils import load_matching_data, load_ocnli_data, load_chid_data, load_wsc_data, \
    load_wplc_data
# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from tools.tokenization_enc_dec import EncDecTokenizer


def get_ltor_prefix_masks_and_position_ids(data,
                                    labels,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,context_lengths):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    # gpt
    # attention_mask = torch.tril(torch.ones(
    #     (att_mask_batch, seq_length, seq_length), device=data.device)).view(
    #         att_mask_batch, 1, seq_length, seq_length)
    # prefix
    attention_mask = torch.zeros(
        (att_mask_batch, seq_length, seq_length), device=data.device).view(
        att_mask_batch, 1, seq_length, seq_length)  # 第二个维度为multichoice的维度

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[labels == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    for b in range(micro_batch_size):

        # Find indecies where EOD token is.
        eod_index = position_ids[b, data[b] == eod_token]
        # Detach indecies from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Set mask for prefix
        attention_mask[b, 0, :context_lengths[b], :context_lengths[b]] = 1
        # attention_mask[b, 0, :, atten_mask_tokens_tensor==1] = 0
        # attention_mask[b, 0, :, 2] = 02
        # attention_mask= torch.where(atten_mask_tokens_tensor[b] > 0, atten_mask_tokens_tensor[b],  attention_mask[b, 0, :, :])
        # attention_mask_np=attention_mask.cpu().numpy()
        # print("")


    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids

def get_batch_loss(context_tokens, label_tokens,second_loss_mask_tokens_tensor,context_lengths):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
    labels = label_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
    second_loss_mask = second_loss_mask_tokens_tensor.view(args.micro_batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    if(args.is_prefix):
        attention_mask, loss_mask, position_ids = get_ltor_prefix_masks_and_position_ids(
            tokens,
            labels,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,context_lengths)
    else:
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            labels,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

    loss_mask[second_loss_mask == 0] = 0.0


    return tokens, attention_mask, loss_mask, position_ids

def cal_output(context_tokens,second_loss_mask,src,group,model):


    # get a sample from batch
    if torch.distributed.get_rank() == 0:
        # raw_text = all_raw_text[input_pos]
        # raw_text_len = len(raw_text)
        # context_tokens = tokenizer.tokenize(raw_text)
        context_length = len(context_tokens)
    else:
        context_length = 0

    # Broadcast length of a sample from rank 0 to others
    input_info = [context_length]
    input_info_tensor = torch.cuda.LongTensor(input_info, device=torch.cuda.current_device())
    torch.distributed.broadcast(input_info_tensor, src, group)
    context_length = input_info_tensor[0].item()

    # Broadcast the sample from rank 0 to others
    if torch.distributed.get_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens,
                                                      device=torch.cuda.current_device())
    else:
        context_tokens_tensor = torch.empty(context_length,
                                            dtype=torch.int64,
                                            device=torch.cuda.current_device())

    torch.distributed.broadcast(context_tokens_tensor, src, group)
    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

    # Compute logits of the one with max possibility and labels
    loss_stream = get_loss_stream(model, [context_tokens], second_loss_mask)
    return loss_stream

def generate_losses_input_from_file(model):

    args = get_args()
    # tokenizer = get_tokenizer()
    tokenizer_decode=EncDecTokenizer(args.vocab_file)
    input_count = 0
    input_pos = 0

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    dataset = {}
    tokenizer = EncDecTokenizer(args.vocab_file)
    if args.task == "ocnli":
        dataset = load_ocnli_data(args, tokenizer)
    elif args.task == "afqmc" or args.task == "bustm" or args.task == "csl" or args.task == "wsc":
    # elif args.task == "afqmc" or args.task == "bustm" or args.task == "csl" :
        dataset = load_matching_data(args, tokenizer)
    elif args.task == "chid":
        dataset = load_chid_data(args, tokenizer)
    # elif args.task == "wsc":
    #     dataset = load_wsc_data(args, tokenizer)
    all_raw_text = dataset["sentence"]
    print("sample_size:{0}".format(len(dataset['sentence'])))
    if torch.distributed.get_rank() == 0:
        # assert dataset!=None, "Error happened when read sample_input_file "
        # all_raw_text = get_sentences(args.sample_input_file)
        input_count = len(all_raw_text)        
        
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('`sample-output-file` not specified, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    # Set source and collection communication group
    src = 0
    group = mpu.get_model_parallel_group()
    
    # Broadcast input_count, and label_class_count from rank 0 to others
    input_count_tensor = torch.cuda.LongTensor([input_count], 
                                device=torch.cuda.current_device())
    torch.distributed.broadcast(input_count_tensor, src, group)
    input_count = input_count_tensor[0].item()


    correct = 0
    total = 0
    pre_1=0
    pre_0=0
    true_1=0

    model.eval()
    with torch.no_grad():
        while True:
            if input_pos == input_count:
                break
            output_=[]
            #print("a_all_raw_text:{0}".format(len(all_raw_text)))
            if(len(all_raw_text)==0 ):
                continue

            # 默认二分类任务
            if(args.task in ["afqmc", "bustm", "csl", "wsc"]):
                output_1=cal_output(all_raw_text[input_pos][0],dataset["second_loss_mask"][input_pos][0],src,group,model)
                output_2=cal_output(all_raw_text[input_pos][1],dataset["second_loss_mask"][input_pos][1],src,group,model)
                # 超过2台节点时，只有第一台节点和最后一台节点上有loss输出，此处指定第一台节点的0号卡为输出；
                if torch.distributed.get_rank() == 0: # 所有卡中的0号卡，如果10个节点/8卡，则是80张卡中的0号卡
                    output_.append(output_1.item())
                    output_.append(output_2.item())
                    res = [np.argmin(np.array(x)) for x in zip([output_[0]], [output_[1]])]
            elif(args.task in ["ocnli"]):
                output_1 = cal_output(all_raw_text[input_pos][0], dataset["second_loss_mask"][input_pos][0], src, group,model)
                output_2 = cal_output(all_raw_text[input_pos][1], dataset["second_loss_mask"][input_pos][1], src, group,model)
                output_3 = cal_output(all_raw_text[input_pos][2], dataset["second_loss_mask"][input_pos][2], src, group,model)
                if torch.distributed.get_rank() == 0:
                    output_.append(output_1.item())
                    output_.append(output_2.item())
                    output_.append(output_3.item())
                    res = [np.argmin(np.array(x)) for x in zip([output_[0]], [output_[1]],[output_[2]])]
            elif(args.task in ["chid"]):
                sample_count=len(all_raw_text[input_pos])
                sample_index=0
                while True:
                    if sample_index == sample_count:
                        break
                    output_3 = cal_output(all_raw_text[input_pos][sample_index], dataset["second_loss_mask"][input_pos][sample_index], src,
                                          group, model)
                    sample_index+=1
                    if torch.distributed.get_rank() == 0:
                        output_.append(output_3.item())
                if torch.distributed.get_rank() == 0:
                    res = [np.argmin(np.array(x)) for x in zip([output_[0]], [output_[1]], [output_[2]], [output_[3]], [output_[4]], [output_[5]], [output_[6]])]
            else:
                print("Error: task_name {0} is not supported".format(args.task))
                return

            #print(output_)
            # Write results to file
            #if torch.distributed.get_rank() == 0:
                # raw_text=tokenizer_decode.decode(context_tokens)
                # print("\nContext:", " ".join(raw_text), flush=True)
                #loss = output_[0].cpu().numpy().tolist()
                #loss = output_.cpu().numpy().tolist()
                #print("\nMegatron-LM:", loss)
                # raw_text = None


            if torch.distributed.get_rank() == 0 and len(dataset["labels"])>0:
            # if mpu.is_pipeline_last_stage():
                labels = dataset["labels"][input_pos]
                pre_1 += sum([x == 1 for x in res])
                pre_0 += sum([x == 0 for x in res])
                res_ = [x == y for x, y in zip(res, [labels])]
                correct += sum(res_)
                total += len(res_)
                true_1 += sum([x == 1 for x in [labels]])
                if total == 0:
                    print("correct:{0},total:{1},acc:{2},pre_1:{3},pre_0:{4},true_1:{5}".format(correct, total, 0, pre_1,pre_0, true_1))
                else:
                    print("correct:{0},total:{1},acc:{2},pre_1:{3},pre_0:{4},true_1:{5}".format(correct, total,correct / total, pre_1, pre_0,true_1))
            if torch.distributed.get_rank() == 0 and len(args.sample_output_file)>0 :
                for res_b in res:
                    pre_dic = {}
                    pre_dic["id"] = input_pos
                    if(type(res_b)==list):
                        res_b=res_b[0] # bs始终为1
                    if args.task in ['bustm',"csl"]:
                        if res_b in ["0","1",0,1] :
                            pre_dic['label'] = str(res_b)
                        else:
                            print("{0} label error".format(args.task))
                    elif args.task == 'wsc':
                        if res_b in ["0",0] :
                            pre_dic['label'] = "false"
                        elif res_b in ["1",1]:
                            pre_dic['label'] = "true"
                        else:
                            print("{0} label error".format(args.task))
                    elif args.task == 'ocnli':
                        if res_b in ["0",  0]:
                            pre_dic['label'] = 'entailment'
                        elif res_b in ["1",  1]:
                            pre_dic['label'] = 'contradiction'
                        elif res_b in ["2",  2]:
                            pre_dic['label'] = 'neutral'
                    elif args.task == 'chid':
                        res_b=int(res_b)
                        pre_dic['answer'] = res_b

                    s = json.dumps(pre_dic, ensure_ascii=False)
                    fname_out.write(s + '\n')

            input_pos += 1


def generate_losses_ppl_input_from_file(model):
    args = get_args()
    # tokenizer = get_tokenizer()
    tokenizer_decode = EncDecTokenizer(args.vocab_file)
    input_count = 0
    input_pos = 0

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    dataset = {}
    tokenizer = EncDecTokenizer(args.vocab_file)
    if args.task == "wplc":
        dataset = load_wplc_data(args, tokenizer)

    all_raw_text = dataset["sentence"]
    print("sample_size:{0}".format(len(dataset['sentence'])))
    if torch.distributed.get_rank() == 0:
        # assert dataset!=None, "Error happened when read sample_input_file "
        # all_raw_text = get_sentences(args.sample_input_file)
        input_count = len(all_raw_text)

        # if args.sample_output_file is None:
        #     sample_output_file = args.sample_input_file + ".out"
        #     print('`sample-output-file` not specified, setting '
        #           'it to {}'.format(sample_output_file))
        # else:
        #     sample_output_file = args.sample_output_file
        # fname_out = open(sample_output_file, "w+")

    # Set source and collection communication group
    src = 0
    group = mpu.get_model_parallel_group()

    # Broadcast input_count, and label_class_count from rank 0 to others
    input_count_tensor = torch.cuda.LongTensor([input_count],
                                               device=torch.cuda.current_device())
    torch.distributed.broadcast(input_count_tensor, src, group)
    input_count = input_count_tensor[0].item()

    model.eval()

    if torch.distributed.get_rank() == 0:
        res = []

    with torch.no_grad():
        while True:
            if input_pos == input_count:
                print("The num of sample:{0}".format(input_count))
                break
            if (len(all_raw_text) == 0):
                continue

            # 默认二分类任务
            if (args.task in ["wplc"]):
                output_1 = cal_output(all_raw_text[input_pos][0], dataset["second_loss_mask"][input_pos][0], src, group,
                                      model)
                # 超过2台节点时，只有第一台节点和最后一台节点上有loss输出，此处指定第一台节点的0号卡为输出；
                if torch.distributed.get_rank() == 0:
                    # for v in output_1.item():
                    #     res.append(v)
                    res.append(output_1.item())
            else:
                print("Error: task_name {0} is not supported".format(args.task))
                return

            input_pos += 1
        if torch.distributed.get_rank() == 0:
            # calculating perplexity
            perplexity = np.exp(res)
            PPL = np.mean(perplexity)
            # PPL_list.append(perplexity.item())
            print('PPL value', PPL)


def pad_batch_loss(batch, pad_id, args):

    context_lengths = []
    labels = []
    tokens_ = []
    tmp_token=[]
    for tokens in batch:
        tmp_token = []
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length + 1))
            labels.append(tokens[1:])
            tokens_.append(tokens[:-1])

            # padding在前
            # tmp_token.extend([pad_id] * (args.seq_length - context_length + 1))
            # tmp_token.extend(tokens)
            # labels.append(tmp_token[1:])
            # tokens_.append(tmp_token[:-1])

        context_lengths.append(context_length)
    return tokens_, labels, context_lengths

        
def get_loss_stream(model, context_tokens,second_loss_mask):

    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens, labels, context_lengths = pad_batch_loss(context_tokens,
                                                tokenizer.eod, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    label_tokens_tensor = torch.cuda.LongTensor(labels)
    second_loss_mask_tokens_tensor = torch.cuda.LongTensor(second_loss_mask)

    torch.distributed.broadcast(context_length_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    torch.distributed.broadcast(label_tokens_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())                                
    torch.distributed.broadcast(second_loss_mask_tokens_tensor,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, loss_mask, position_ids = get_batch_loss(context_tokens_tensor, label_tokens_tensor,second_loss_mask_tokens_tensor,context_lengths)

    batch_loss = sample_loss_batch(model, 
                            context_tokens_tensor,
                            context_length_tensor,
                            label_tokens_tensor,
                            attention_mask,
                            loss_mask,
                            position_ids)

    if tokens is not None:
        return batch_loss
    else:
        return None



def forward_step_loss(model, tokens, labels, position_ids, attention_mask, tokentype_ids,
                 layer_past=None, get_key_value=None,
                 forward_method_parallel_output=None):

    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor = model(tokens,
                          position_ids,
                          attention_mask,
                          labels=labels,
                          tokentype_ids=tokentype_ids,
                          layer_past=layer_past,
                          get_key_value=get_key_value,
                          forward_method_parallel_output=forward_method_parallel_output)

    if get_key_value:
        output_tensor, layer_past = output_tensor

    send_forward(output_tensor)

    args.seq_length = orig_seq_length
    if get_key_value:
        return output_tensor, layer_past
    return output_tensor


def apply_loss_mask(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

def sample_loss_batch(model,
                      context_tokens,
                      context_lengths,
                      label_class_tokens,
                      attention_mask,
                      loss_mask,
                      position_ids,
                      maxlen=None, type_ids=None):

    args = get_args()
   
    model.eval()
    with torch.no_grad():
    
        context_length = context_lengths.min().item()

        layer_past = None
        batch_size = context_tokens.size(0)        
        tokens = context_tokens
        labels = label_class_tokens
        
        loss = torch.empty((batch_size, 1),
                        dtype = torch.float,
                        device = tokens.device)         
        
        if args.recompute:
            output = forward_step_loss(model, tokens, labels,
                                      position_ids,
                                      attention_mask,
                                      tokentype_ids=type_ids,
                                      forward_method_parallel_output=args.parallel_output)
            if mpu.is_pipeline_last_stage():
                assert output is not None
                # compute loss with loss_mask

                loss = apply_loss_mask(loss_mask, output)


        else:
            assert False, "Donot support other modes than recompute. Please take --recompute to recompute all the attentions"

        if mpu.is_pipeline_last_stage():
            new_tokens = loss
            src = mpu.get_pipeline_model_parallel_last_rank()
            group = mpu.get_embedding_group()
            torch.distributed.broadcast(new_tokens, src, group)
            return loss

        else:
            if mpu.is_pipeline_first_stage():
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_embedding_group()
                new_tokens = torch.empty_like(loss)
                torch.distributed.broadcast(new_tokens, src, group)
                loss = new_tokens
                return loss
            else:
                return None



