#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import json
from pathlib import Path

import torch

from .data import pad_sents


SPECIAL_TOKENS = ["<pad>", "<unk>", "<mask>"]
REQUIRED_SPECIAL_TOKENS = ["<pad>", "<unk>"]


class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id is None:
            word2id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

        self.word2id = dict(word2id)
        self._validate_special_tokens()
        self.pad_id = self.word2id["<pad>"]
        self.unk_id = self.word2id["<unk>"]
        self.mask_id = self.word2id.get("<mask>")
        self.id2word = {v: k for k, v in self.word2id.items()}

    def _validate_special_tokens(self):
        for idx, token in enumerate(REQUIRED_SPECIAL_TOKENS):
            if token not in self.word2id:
                raise ValueError(f"vocabulary is missing required token {token!r}")
            if self.word2id[token] != idx:
                raise ValueError(f"vocabulary token {token!r} must have id {idx}")

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents, device):
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size=None, freq_cutoff=1):
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))

        top_k_words = sorted(valid_words, key=lambda w: (-word_freq[w], w))
        if size is not None:
            top_k_words = top_k_words[:size]

        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

    @staticmethod
    def from_subword_list(subword_list):
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Vocab(object):
    def __init__(self, vocab):
        self.vocab = vocab

    @staticmethod
    def build(sents, size=None, freq_cutoff=1):
        print('initialize vocabulary ..')
        vocab = VocabEntry.from_corpus(sents, size=size, freq_cutoff=freq_cutoff)
        return Vocab(vocab)

    def save(self, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(self.vocab.word2id, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf8') as f:
            entry = json.load(f)

        if "src_word2id" in entry:
            word2id = entry["src_word2id"]
        else:
            word2id = entry

        return Vocab(VocabEntry(word2id))

    def __repr__(self):
        return 'Vocab(%d words)' % len(self.vocab)
