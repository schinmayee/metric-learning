#!/usr/bin/env bash
./train.py --num-train 4 --num-val 2 --num-test 4 --triplets-per-class 8 \
    --network Inception --lr 5e-4 --feature-size 64 --reg 1e-4 --epochs 12 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16
