#!/usr/bin/env bash
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-4 --feature-size 64 --reg 1e-4 --epochs 8 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss SquareHingeL2 --mining Hardest \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-6 --feature-size 64 --reg 1e-3 --epochs 8 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss SquareHingeL2 --mining Hardest \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-3 --feature-size 64 --reg 1e-4 --epochs 8 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss SquareHingeL2 --mining Hardest \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 2e-5 --feature-size 64 --reg 5e-5 --epochs 8 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss SquareHingeL2  --mining Hardest \
    --normalize-features
