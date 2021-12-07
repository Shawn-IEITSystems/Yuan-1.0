#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:shenchong
@file:generate_loss_gpt_utils.py
@time:2021/09/03
"""
import os
import torch
import json
import torch
import numpy as np
from megatron import mpu, print_rank_0, get_args
import random
from tools.tokenization_enc_dec import EncDecTokenizer

def load_fewshot_data(args):
    filename = os.path.join(args.fewshot_path)
    objs = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            if(len(line.strip())<2):
                continue
            objs.append(line.strip())
    return "".join(objs)

def load_json_data(args, tokenizer):
    # filename = os.path.join(args.sample_input_file, data_type+'.json')
    filename = os.path.join(args.sample_input_file)
    objs = []
    with open(filename,'r',encoding='utf-8') as fin:
        for line in fin:
            if(len(line.strip())<1):
                continue
            objs.append(json.loads(line.strip()))

    pad_id = tokenizer.encoder['<pad>']   #固定长度
    args.eod_token = tokenizer.encoder['<eod>'] #每个token的结尾
    return args, pad_id, objs

def trans_tokens(args, pad_id, prompt, tokenizer, sentence):
    if prompt == None:
        prompt_len = 1
        tokens = tokenizer.encode(sentence)
    else:
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)
        tokens = prompt_tokens + tokenizer.encode(sentence)[:args.seq_length - prompt_len]
    # second_mask = [0] * (args.seq_length - 1)
    second_mask = [0] * (args.seq_length)
    # for idx in range(prompt_len - 1, len(tokens) - 1): # loss mask,计算除prompt_len以外token的loss
    #    second_mask[idx] = 1


    if(args.task=="wplc"):
        # padding 在前args.seq_length-len(tokens)
        # for idx in range(0,args.seq_length - len(tokens) + len(tokens) - 1):  # loss mask,计算除prompt_len以外token的loss
        #     second_mask[idx] = 1
        # padding 在后 取所有输入的loss
        for idx in range(0, len(tokens) - 1):
           second_mask[idx] = 1
    else:
        # padding 在前args.seq_length-len(tokens)
        # for idx in range(args.seq_length-len(tokens)+prompt_len - 1, args.seq_length-len(tokens)+len(tokens) - 1): # loss mask,计算除prompt_len以外token的loss
        #    second_mask[idx] = 1
        # padding 在后
        for idx in range(prompt_len - 1, len(tokens) - 1):  # loss mask,计算除prompt_len以外token的loss
           second_mask[idx] = 1

    # for idx in range(len(tokens) - 1): # loss mask ，计算所有token的loss
    #     second_mask[idx] = 1
    token_length = len(tokens)
    assert token_length < args.seq_length
    #print(args.seq_length)
    # tokens.extend([pad_id] * (args.seq_length - token_length)) # 补全
    return second_mask, tokens


def load_matching_data(args, tokenizer):
    args, pad_id, objs = load_json_data(args, tokenizer)

    if(args.is_fewshot):
        fewshot_prefex=load_fewshot_data(args)
        print("fewshot_prefex:{0}".format(fewshot_prefex))
    else:
        fewshot_prefex=""

    all_labels = []

    i = 0
    end1 = ""
    end2 = ""
    # end1='。矛盾或一致：矛盾'
    # end2='。矛盾或一致：一致'

    # print('train_len: ',len(train_objs))
    # random.seed(1)
    # rand_index=random.randint(0,len(train_objs)-1)
    # rand_index=0

    dataset={"sentence":[],"second_loss_mask":[],"labels":[], "ids": []}

    for obj in objs:

        if args.task == "csl":
            # p_shot='没希望了？错，难道还有希望吗。我厉害不厉害？对，我牛逼不牛逼。'
            keyword = ','.join(obj['keyword'])
            prompt1 = fewshot_prefex+"{}的关键词不是，".format(obj['abst'])
            prompt2 = fewshot_prefex+"{}的关键词是，".format(obj['abst'])
            sentence1 = obj['abst']
        elif args.task == "wsc":
            if 'label'in obj and obj['label'] == '-':
                continue
            keyword = obj['target']['span1_text']
            prompt1 = fewshot_prefex+obj['text'] + "其中第{}个字的{}是".format(obj['target']['span2_index'], obj['target']['span2_text'])
            prompt2 = fewshot_prefex+obj['text'] + "其中第{}个字的{}不是".format(obj['target']['span2_index'], obj['target']['span2_text'])
            sentence1 = obj['text']
        else:
            # test
            # prompt1='话说有时候我就有点难过，错，'
            # prompt2='话说有时候我就有点难过，对，'
            # keyword='有时候我就有点难过'
            # sentence1='话说有时候我就有点难过'
            #
            if 'label'in obj and obj['label'] == '-':
                continue
            keyword = obj['sentence2']
            prompt1 = fewshot_prefex+"{}？错，".format(obj['sentence1'])
            prompt2 = fewshot_prefex+"{}？对，".format(obj['sentence1'])

            # rand_index=random.randint(0,len(train_objs)-1)
            # if(train_objs[rand_index]['label'] in ['0',0]):
            #    p_shot=train_objs[rand_index]['sentence1']+'？错，'+train_objs[rand_index]['sentence2']+'。'
            # else:
            #    p_shot=train_objs[rand_index]['sentence1']+'？对，'+train_objs[rand_index]['sentence2']+'。'

            # print(p_shot)
            # p_shot='没希望了？错，难道还有希望吗。我厉害不厉害？对，我牛逼不牛逼。'
            # p_shot='我厉害不厉害？对，我牛逼不牛逼。'
            # p_shot='没希望了？错，难道还有希望吗。'

            # prompt1 =p_shot+ "{}？错，".format(obj['sentence1'])
            # prompt2 =p_shot+ "{}？对，".format(obj['sentence1'])

            # prompt1 = "{}。".format(obj['sentence1'])
            # prompt2 = "{}。".format(obj['sentence1'])
            # sentence1=obj['sentence1']

        second_loss_mask1, tokens1 = trans_tokens(args, pad_id, prompt1, tokenizer, keyword + end1)
        second_loss_mask2, tokens2 = trans_tokens(args, pad_id, prompt2, tokenizer, keyword + end2)
        i += 1
        if (i == 1):
            print(prompt1 + keyword + end1)
            print(prompt2 + keyword + end2)
        if 'label'in obj:
            if args.task == "wsc":
                if obj['label'] == 'true':
                    all_labels.append([0])
                elif obj['label'] == 'false':
                    all_labels.append([1])
            else:
                all_labels.append(int(obj['label']))

        dataset["sentence"].append([tokens1, tokens2])
        dataset["second_loss_mask"].append([second_loss_mask1, second_loss_mask2])
        if("id" in obj):
            dataset["ids"].append(obj["id"])
    dataset["labels"]=all_labels

    return dataset

def load_ocnli_data(args, tokenizer):
    args, pad_id, objs = load_json_data(args, tokenizer)
    if(args.is_fewshot):
        fewshot_prefex=load_fewshot_data(args)
        print("fewshot_prefex:{0}".format(fewshot_prefex))
    else:
        fewshot_prefex=""

    all_labels = []
    dataset = {"sentence": [], "second_loss_mask": [], "labels": [],"ids":[]}
    i=0
    for obj in objs:
        if 'label'in obj and obj['label'] == '-':
            continue

        prompt = fewshot_prefex+"{}？对，".format(obj['sentence1'])
        second_mask1, tokens1 = trans_tokens(args, pad_id, prompt, tokenizer, obj['sentence2'])

        prompt = fewshot_prefex+"{}？错，".format(obj['sentence1'])
        second_mask2, tokens2 = trans_tokens(args, pad_id, prompt, tokenizer, obj['sentence2'])

        prompt = fewshot_prefex+"{}？也许，".format(obj['sentence1'])
        second_mask3, tokens3 = trans_tokens(args, pad_id, prompt, tokenizer, obj['sentence2'])
        i += 1
        if (i == 1):
            print(prompt + obj['sentence2'] )

        if 'label' in obj:
            if obj['label'] == 'entailment':
                all_labels.append(0)
            elif obj['label'] == 'contradiction':
                all_labels.append(1)
            else:
                all_labels.append(2)


        dataset["sentence"].append([tokens1, tokens2, tokens3])
        dataset["second_loss_mask"].append([second_mask1, second_mask2, second_mask3])
        if ("id" in obj):
            dataset["ids"].append(obj["id"])
    dataset["labels"] = all_labels

    return dataset

def load_chid_data(args,  tokenizer):
    args, pad_id, objs = load_json_data(args, tokenizer)
    if(args.is_fewshot):
        fewshot_prefex=load_fewshot_data(args)
        print("fewshot_prefex:{0}".format(fewshot_prefex))
    else:
        fewshot_prefex=""
    all_labels = []

    dataset = {"sentence": [], "second_loss_mask": [], "labels": [],"ids":[]}
    i=0
    for _, obj in enumerate(objs):
        if 'answer' in obj:
            all_labels.append(int(obj['answer']))
        s_tokens = []
        s_masks = []
        # 每个标签都对应一次token
        for _, label in enumerate(obj['candidates']):
            sentence = fewshot_prefex+obj['content'].replace('#idiom#', label)
            second_mask, tokens = trans_tokens(args, pad_id, None, tokenizer, sentence)
            s_masks.append(second_mask)
            s_tokens.append(tokens)
            if (i == 0):
                print_rank_0("".join(sentence))
            i += 1
        dataset["sentence"].append(s_tokens)
        dataset["second_loss_mask"].append(s_masks)
        if ("id" in obj):
            dataset["ids"].append(obj["id"])
    dataset["labels"] = all_labels

    return dataset

def load_wsc_data(args,  tokenizer):
    args, pad_id, objs = load_json_data(args, tokenizer)

    all_labels = []

    dataset = {"sentence": [], "second_loss_mask": [], "labels": []}
    i=0
    for _, obj in enumerate(objs):
        true_ans=obj["target"]["span1_text"]

        # all_labels.append(int(obj['answer']))
        s_tokens = []
        s_masks = []
        # 每个标签都对应一次token
        for _, cand_ans in enumerate(obj['candidates']):
            sentence = obj['text']
            span2_index=obj["target"]["span2_index"]
            span2_len=len(obj["target"]["span2_text"])
            ori_text=sentence[:span2_index]+cand_ans+sentence[span2_index+span2_len:]
            second_mask, tokens = trans_tokens(args, pad_id, None, tokenizer, ori_text)
            s_masks.append(second_mask)
            s_tokens.append(tokens)
            if (i == 0):
                print_rank_0("\n".join(sentence))
            i += 1
        dataset["sentence"].append(s_tokens)
        dataset["second_loss_mask"].append(s_masks)
        true_ans_index=obj['candidates'].index(true_ans)
        all_labels.append(true_ans_index)

    dataset["labels"] = all_labels

    return dataset

def load_wsc_data_test():
    objs=[]
    i=0
    with open("/mnt/shenchong/Megatron-LM/data/dev-yt.json",'r',encoding='utf-8') as fin:
        for line in fin:
            if(len(line.strip())<1):
                continue
            objs.append(json.loads(line.strip()))

    all_labels = []

    dataset = {"sentence": [], "second_loss_mask": [], "labels": []}
    i=0
    for _, obj in enumerate(objs):
        true_ans=obj["target"]["span1_text"]

        # all_labels.append(int(obj['answer']))
        s_tokens = []
        s_masks = []
        # 每个标签都对应一次token
        for _, cand_ans in enumerate(obj['candidates']):
            sentence = obj['text']
            span2_index=obj["target"]["span2_index"]
            span2_len=len(obj["target"]["span2_text"])
            ori_text=sentence[:span2_index]+cand_ans+sentence[span2_index+span2_len:]
            print("{0},{1},{2}".format(obj["target"]["span1_text"],obj["target"]["span2_text"],ori_text))
        true_ans_index=obj['candidates'].index(true_ans)
        all_labels.append(true_ans_index)

    dataset["labels"] = all_labels

    return dataset

def load_wplc_data(args, tokenizer):
    args, pad_id, objs = load_json_data(args, tokenizer)

    all_tokens = []
    all_masks = []
    dataset = {"sentence": [], "second_loss_mask": [], "labels": []}

    for _, obj in enumerate(objs):
        text1 = obj['masked_text'].split('<mask>')[0].strip()
        text2 = obj['masked_text'].split('<mask>')[-1].strip()
        correct_word = obj['correct_word'].strip()
        sentence = text1 + correct_word + text2

        second_mask, tokens = trans_tokens(args, pad_id, None, tokenizer, sentence)
        all_masks.append([second_mask])
        all_tokens.append([tokens])

    #all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    #all_masks = torch.tensor(all_masks, dtype=torch.float)
    #dataset = TensorDataset(all_tokens, all_masks)
    dataset["sentence"] = all_tokens
    dataset["second_loss_mask"] = all_masks
    # Torch dataloader.
    return dataset


if __name__ == '__main__':

    # args = get_args()
    # tokenizer = EncDecTokenizer(args.vocab_file)
    # load_wsc_data(args, tokenizer)
    load_wsc_data_test()

    pass
