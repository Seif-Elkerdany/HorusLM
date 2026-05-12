#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


MASK_TOKEN = "<mask>"


def pad_sents(sents, pad_token):
    sents_padded = []

    max_len = max([len(sent) for sent in sents])
    for sent in sents:
        sents_padded.append(sent + [pad_token] * (max_len - len(sent)))

    return sents_padded


def read_sentences(file_path):
    sents = []

    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sents.append(line.split())

    return sents


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=len, reverse=True)
        yield examples


def mask_sentence(sent, position, mask_token = MASK_TOKEN):
    masked_sent = list(sent)
    target = masked_sent[position]
    masked_sent[position] = mask_token
    return masked_sent, target


def make_random_masked_batch(sents, mask_token = MASK_TOKEN, masks_per_sentence = 1):
    if masks_per_sentence < 1:
        raise ValueError("masks_per_sentence must be at least 1")

    masked_sents = []
    mask_positions = []
    targets = []

    for sent in sents:
        if not sent:
            continue

        num_masks = min(masks_per_sentence, len(sent))
        positions = np.random.choice(len(sent), size=num_masks, replace=False)
        for position in positions:
            masked_sent, target = mask_sentence(sent, position, mask_token)
            masked_sents.append(masked_sent)
            mask_positions.append(position)
            targets.append(target)

    return masked_sents, mask_positions, targets


def iter_all_masked_batches(
    sents,
    batch_size,
    mask_token = MASK_TOKEN,
):
    masked_sents = []
    mask_positions = []
    targets = []

    for sent in sents:
        for position in range(len(sent)):
            masked_sent, target = mask_sentence(sent, position, mask_token)
            masked_sents.append(masked_sent)
            mask_positions.append(position)
            targets.append(target)

            if len(masked_sents) == batch_size:
                yield masked_sents, mask_positions, targets
                masked_sents = []
                mask_positions = []
                targets = []

    if masked_sents:
        yield masked_sents, mask_positions, targets
