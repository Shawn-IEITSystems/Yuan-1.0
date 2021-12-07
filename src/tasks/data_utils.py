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

""" Tasks data utility."""

import re
import numpy as np
from megatron import get_args

def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    for _ in range(3):
        text = text.replace(' . ', '. ')

    return text


def build_sample(ids, paddings, label, unique_id):
    """Convert to numpy and return a sample consumed by the batch producer."""

    ids_np = np.array(ids, dtype=np.int64)
    paddings_np = np.array(paddings, dtype=np.int64)
    sample = ({'text': ids_np,
               'padding_mask': paddings_np,
               'label': int(label),
               'uid': int(unique_id)})

    return sample


def build_tokens_types_paddings_from_text(text_a, text_b,
                                          tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    text_a_ids = tokenizer.tokenize(text_a)
    text_b_ids = None
    if text_b is not None:
        text_b_ids = tokenizer.tokenize(text_b)

    return build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids,
                                                max_seq_length, tokenizer.eod)


def build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids, max_seq_length,
                                         eod_id):
    """Build token types and paddings, trim if needed, and pad if needed."""

    ids = []
    paddings = []
    args = get_args()

    # A.
    # len_text_a = len(text_a_ids)
    # ids.extend(text_a_ids)
    # paddings.extend([1] * len_text_a)
    
    len_text_a = len(text_a_ids)
    if text_b_ids is not None:
        len_text_b = len(text_b_ids)
    else:
        len_text_b = 0
    text_a_ids = text_a_ids[:min(args.seq_length-len_text_b-2, len_text_a)]
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    paddings.extend([1] * len_text_a)

    # # [SEP].
    # ids.append(4)
    # paddings.append(1)
    
    # B.
    if text_b_ids is not None:
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        paddings.extend([1] * len_text_b)

    # Cap the size.
    trimmed = False
    if len(ids) >= max_seq_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
        trimmed = True

    # <eod>.
    if (text_b_ids is not None) or trimmed:
        ids.append(eod_id)
        paddings.append(1)

    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([5] * padding_length)
        paddings.extend([0] * padding_length)

    return ids, paddings
