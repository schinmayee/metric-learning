#!/usr/bin/env bash
./train.py --num-train 2 --num-val 2 --num-test 2 --triplets-per-class 4 \
    --network Inception --lr 5e-4 --feature-size 16 --reg 1e-4 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 8 --no-cuda
