#!/usr/bin/env bash
./train_pushed.py --num-train 80 --num-val 40 --num-test 80 --batch-size 64\
    --network Inception --lr 4e-5 --feature-size 64 --reg 5e-3 --epochs 2 \
    --results-freq 1 --test-results-freq 1 --val-freq 1 --num-batches 6\
    --normalize-features
