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

"""CHID dataset."""
import glob
import json
import os
import time

from megatron import print_rank_0, get_args
from torch.utils.data import Dataset
from tasks.data_utils import build_sample
from tasks.data_utils import build_tokens_types_paddings_from_ids
from tasks.data_utils import clean_text
from tqdm import tqdm


NUM_CHOICES=7


class CHIDDataset(Dataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length):

        self.dataset_name = name

        if(name=="training"): # train or eval
            self.t_e=1
        else:
            self.t_e=0
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(process_single_datapath(datapath, tokenizer, max_seq_length))
        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))
        self.sample_multiplier = NUM_CHOICES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def process_single_datapath(filename, tokenizer, max_seq_length):
    """"Implement abstract method."""
    args = get_args()

    print_rank_0(' > Processing {} ...'.format(filename))
    start_time = time.time()

    samples = []

    num_samples=0

    with open(filename, 'r',encoding='utf-8') as f:
        lines_iter = tqdm(f.readlines(), disable=False) if args.local_rank == 0 else f.readlines()
        for line in lines_iter:
            ids_list = []
            paddings_list = []
            d = json.loads(line)
            label = d["answer"]
            candidates = d["candidates"]
            assert label >= 0
            assert label < NUM_CHOICES

            for c in candidates:
                text_a = clean_text(d["content"].strip().replace(" ","").replace("...",""))

                if text_a[-1]==".":
                    text_a = text_a[:-1] +"。"
                elif text_a[-1] not in ["?","!","。","？","！",":","：",";","；",",","，",":","："]:
                    text_a+="。"
                text_a = text_a.replace('#idiom#', c)
                text_a_ids = tokenizer.tokenize(text_a)

                qa_ids = []
                ids, paddings = build_tokens_types_paddings_from_ids(
                                text_a_ids, qa_ids, max_seq_length,
                                    tokenizer.eod)
                ids_list.append(ids)
                paddings_list.append(paddings)
            samples.append(build_sample(ids_list, paddings_list, label, num_samples))
            num_samples += 1
    elapsed_time = time.time() - start_time
    print_rank_0('    > processed {} samples in {:.2f} seconds'.format(num_samples, elapsed_time))
    return samples


class CHIDTestDataset(Dataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length):

        self.dataset_name = name

        if(name=="training"): # train or eval
            self.t_e=1
        else:
            self.t_e=0
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(process_test_datapath(datapath, tokenizer, max_seq_length))
        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))
        self.sample_multiplier = NUM_CHOICES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def process_test_datapath(filename, tokenizer, max_seq_length):
    """"Implement abstract method."""
    args = get_args()

    print_rank_0(' > Processing {} ...'.format(filename))
    start_time = time.time()

    samples = []
    num_samples=0

    with open(filename, 'r',encoding='utf-8') as f:
        lines_iter = tqdm(f.readlines(), disable=False) if args.local_rank == 0 else f.readlines()
        for line in lines_iter:
            ids_list = []
            paddings_list = []
            d = json.loads(line)

            candidates = d["candidates"]

            for c in candidates:
                text_a = clean_text(d["content"].strip().replace(" ","").replace("...",""))

                if text_a[-1]==".":
                    text_a = text_a[:-1] +"▒~@~B"
                elif text_a[-1] not in ["?","!","▒~@~B","▒~_","▒~A",":","▒~Z",";","▒~[",",","▒~L",":","▒~Z"]:
                    text_a+="▒~@~B"
                text_a = text_a.replace('#idiom#', c)
                text_a_ids = tokenizer.tokenize(text_a)
                qa_ids = []
                ids, paddings \
                        = build_tokens_types_paddings_from_ids(
                                text_a_ids, qa_ids, max_seq_length,
                                    tokenizer.eod)
                ids_list.append(ids)
                paddings_list.append(paddings)
            samples.append(build_sample(ids_list, paddings_list, -1, num_samples))
            num_samples += 1
    elapsed_time = time.time() - start_time
    print_rank_0('    > processed {} samples in {:.2f} seconds'.format(num_samples, elapsed_time))
    return samples
