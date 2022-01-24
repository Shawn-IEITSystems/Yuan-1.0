#-*- coding : utf-8-*-
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

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

#from data_utils.tokenization_enc_dec import EncDecTokenizer
from tokenization_enc_dec import EncDecTokenizer
from megatron.data import indexed_dataset

# from ray.util.multiprocessing.pool import Pool

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = EncDecTokenizer(os.path.join(self.args.vocab_path, 'vocab.txt'))

        Encoder.splitter = IdentitySplitter()

    def encode(self, line):
        # end with <eod>
        if len(line) > 20000:
            return None,  0
        if len(line) < 10:
            return None,  0
        data = line.strip()
        data=line.strip().strip('<n>')
        data = data.replace("<n>","\n")
        if not self.args.sentence_splitter:
            doc_ids = Encoder.tokenizer.encode(data)
            doc_ids.append(Encoder.tokenizer.eod_id)
        else:
            data = data.split(self.args.sentence_splitter)
            data = [item.strip() for item in data]
            data = [ item for item in data if item]
            doc_ids = [ Encoder.tokenizer.encode(item) for item in data]
            doc_ids[-1].append(Encoder.tokenizer.eod_id)

        return doc_ids, len(line)

    def random_spans_noise_mask(self, length, noisy_density=0.15, mean_noise_span_length=10.0):
        num_noise_tokens = round(length * noisy_density)
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def random_segment(seq_length, num_segment):
            x = (torch.arange(seq_length - 1) < (num_segment - 1)).long()
            a = torch.randperm(seq_length - 1, generator=g)
            x = x[a]
            x = F.pad(x, [1, 0])
            segment_id = torch.cumsum(x, dim=0)
            segment_lengths = torch.zeros(num_segment, dtype=torch.long).scatter_add_(0, segment_id, torch.ones(seq_length, dtype=torch.long))

            return segment_lengths

        noise_span_lengths = random_segment(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segment(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = torch.stack([nonnoise_span_lengths, noise_span_lengths], dim=1).view(num_noise_spans * 2)
        span_start_ends = torch.cumsum(interleaved_span_lengths, dim=0).view(-1, 2)
        return span_start_ends.tolist()


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="/mnt/inspur/zhaoxudong/dataset/poem_orig/", type=str, help='Path to input TXT')
    
    group = parser.add_argument_group(title='vocab path')
    group.add_argument('--vocab_path', default="./", type=str, help='Path of vocab_file')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="/workspace/data/asc/", type=str)
    group.add_argument('--output_prefix', default="asc", type=str,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=2,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--sentence_splitter',type=str, default=None)
    group.add_argument('--mean_noise_span_length', type=int, default=3)

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

def getfiles(path,ex_str='py'):
    file_list=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if(len(ex_str)>0 and ex_str in name ):
                continue
            file_list.append(os.path.join(root,name))
    return file_list

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin_list=getfiles(args.input)
    
    for fin_path in fin_list:

        if not os.path.exists(fin_path):
            continue
        print(fin_path)
        fin = open(fin_path, 'r', encoding='utf-8',errors='ignore')

        encoder = Encoder(args)
        tokenizer = EncDecTokenizer(os.path.join(args.vocab_path, 'vocab.txt'))
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

        # use the tokenizer to encode the sentences
        encoded_docs = pool.imap_unordered(encoder.encode, fin, 30)

        if args.sentence_splitter:
            level = "sentence"
        else:
            level = "document"

        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Output prefix: {args.output_prefix}")

        fin_name = fin_path.split('/')[-1]
        context_bin_file = os.path.join(args.output_path, "{}_{}_context.bin".format(fin_name, level))
        context_idx_file = os.path.join(args.output_path,  "{}_{}_context.idx".format(fin_name, level))

        if os.path.exists(context_idx_file):
            continue
        builder_context = indexed_dataset.make_builder(context_bin_file, impl=args.dataset_impl)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        print("tokenizer vocab size:", tokenizer.vocab_size)
        total_tokens=0
        for i, (no_noise_tokens, bytes_processed) in enumerate(encoded_docs, start=1):
            if no_noise_tokens is None :
                continue
            total_tokens+=len(no_noise_tokens)
            total_bytes_processed += bytes_processed

            if level == "document":
                builder_context.add_item(torch.IntTensor(no_noise_tokens))
            if level == "sentence":
                for key, sentence in enumerate(no_noise_tokens):
                    if len(sentence) == 0:
                        continue
                    #for sentence in sentences:
                    builder_context.add_item(torch.IntTensor(sentence))
                builder_context.end_document()

            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {i} documents",
                      f"({i/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)

        builder_context.finalize(context_idx_file)
        print("Total time to used:", time.time() - startup_start)
        pool.close()
        print("total tokens: ",total_tokens )

if __name__ == '__main__':
    main()

