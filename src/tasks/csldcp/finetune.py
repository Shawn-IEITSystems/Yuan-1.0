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

"""CLUE finetuning/evaluation."""

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.model.multiple_choice_for_gpt import MultipleChoiceForGPT
from tasks.eval_utils import accuracy_func_provider, predict_test_func_provider
from tasks.finetune_utils import finetune
from megatron import get_timers
import random
import torch
from functools import partial
from megatron.utils import average_losses_across_data_parallel_group
from sklearn import metrics
from tasks.csldcp.csldcp_dataset import CSLDCPDataset, CSLDCPTestDataset


def train_valid_datasets_provider():
    """Provide train and validation datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    train_dataset = CSLDCPDataset('training', args.train_data,
                                tokenizer, args.seq_length)
    valid_dataset = CSLDCPDataset('validation', args.valid_data,
                                tokenizer, args.seq_length)

    return train_dataset, valid_dataset

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    print_rank_0('building classification model for {} ...'.format(
        args.task))
    model = MultipleChoiceForGPT(num_tokentypes=0,
                           parallel_output=False,
                           pre_process=pre_process,
                           post_process=post_process)

    return model


def metrics_func_provider():
    """Privde metrics callback function."""
    args = get_args()
    tokenizer = get_tokenizer()
    def single_dataset_provider(datapath):
        name = 'csldcp accuracy'
        return CSLDCPDataset(name, [datapath], tokenizer, args.seq_length)
    return accuracy_func_provider(single_dataset_provider)

def predict_func_provider():
    """Privde metrics callback function."""
    args = get_args()
    tokenizer = get_tokenizer()
    def single_dataset_provider(datapath):
        name = 'csldcp test data predict'
        return CSLDCPTestDataset(name, [datapath], tokenizer, args.seq_length)
    return predict_test_func_provider(single_dataset_provider)

def main():
     finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider, predict_callback_provider=predict_func_provider)
