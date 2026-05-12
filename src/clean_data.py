#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from .data import read_sentences
from .seed import set_seed
from .vocab import Vocab


def read_lines(path):
    with open(path, "r", encoding="utf8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    with open(path, "w", encoding="utf8") as f:
        for line in lines:
            f.write(line + "\n")


def unique_lines(lines):
    unique = []
    seen = set()
    duplicates = 0

    for line in lines:
        if line in seen:
            duplicates += 1
            continue
        seen.add(line)
        unique.append(line)

    return unique, duplicates


def remove_overlaps(lines, blocked_lines):
    cleaned = []
    removed = 0
    blocked_lines = set(blocked_lines)

    for line in lines:
        if line in blocked_lines:
            removed += 1
            continue
        cleaned.append(line)

    return cleaned, removed


def clean_dataset(dataset, data_dir="data", output_dataset=None):
    data_root = Path(data_dir)
    input_dir = data_root / dataset
    if output_dataset is None:
        output_dataset = dataset + "_clean"
    output_dir = data_root / output_dataset

    train_path = input_dir / "train.txt"
    val_path = input_dir / "val.txt"
    test_path = input_dir / "test.txt"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_lines = read_lines(train_path)
    val_lines = read_lines(val_path)
    test_lines = read_lines(test_path)

    unique_train, train_duplicates = unique_lines(train_lines)
    unique_val, val_duplicates = unique_lines(val_lines)
    unique_test, test_duplicates = unique_lines(test_lines)

    cleaned_train, train_overlap = remove_overlaps(unique_train, set(unique_val) | set(unique_test))
    cleaned_val, val_overlap = remove_overlaps(unique_val, set(cleaned_train))
    cleaned_test, test_train_overlap = remove_overlaps(unique_test, set(cleaned_train))
    cleaned_test, test_val_overlap = remove_overlaps(cleaned_test, set(cleaned_val))

    write_lines(output_dir / "train.txt", cleaned_train)
    write_lines(output_dir / "val.txt", cleaned_val)
    write_lines(output_dir / "test.txt", cleaned_test)

    train_sents = read_sentences(output_dir / "train.txt")
    Vocab.build(train_sents).save(output_dir / "vocab.json")

    print("input dataset:", input_dir)
    print("output dataset:", output_dir)
    print("original train lines:", len(train_lines))
    print("cleaned train lines:", len(cleaned_train))
    print("removed duplicate train lines:", train_duplicates)
    print("removed train lines overlapping val/test:", train_overlap)
    print("original val lines:", len(val_lines))
    print("cleaned val lines:", len(cleaned_val))
    print("removed duplicate val lines:", val_duplicates)
    print("removed val lines overlapping train:", val_overlap)
    print("original test lines:", len(test_lines))
    print("cleaned test lines:", len(cleaned_test))
    print("removed duplicate test lines:", test_duplicates)
    print("removed test lines overlapping train:", test_train_overlap)
    print("removed test lines overlapping val:", test_val_overlap)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Clean SeshatLM data for masked hieroglyph prediction")
    parser.add_argument("--dataset", required=True, help="dataset folder under data/, for example aes")
    parser.add_argument("--data_dir", default="data", help="root data directory")
    parser.add_argument("--output_dataset", default=None, help="output dataset folder name")
    return parser.parse_args(argv)


def main(argv=None):
    set_seed()
    args = parse_args(argv)
    clean_dataset(args.dataset, data_dir=args.data_dir, output_dataset=args.output_dataset)


if __name__ == "__main__":
    main()
