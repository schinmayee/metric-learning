#!/usr/bin/env bash
./train_pushed.py --num-train 120 --num-val 40 --num-test 80 --batch-size 64\
    --network Inception --lr 4e-5 --feature-size 64 --reg 1e-3 --epochs 500 \
    --results-freq 1 --val-freq 1 --num-batches 10\
    --normalize-features
