#!/usr/bin/env bash
LOSS=SquareHingeL2
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network Inception --lr 4e-5 --feature-size 64 --reg 5e-3 --epochs 3 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 32  --loss ${LOSS} --mining SemiHard \
    --normalize-features
