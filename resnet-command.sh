#!/usr/bin/env bash
LOSS=HingeL2
#LOSS=SquareHingeL2
#LOSS=Ratio
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-5 --feature-size 64 --reg 1e-4 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS} --mining Hardest \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-7 --feature-size 64 --reg 1e-3 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS} --mining Hardest \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 1e-5 --feature-size 64 --reg 5e-4 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS}  --mining Hardest \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-5 --feature-size 64 --reg 1e-4 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS} --mining SemiHard \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-7 --feature-size 64 --reg 1e-3 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS} --mining SemiHard \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 1e-5 --feature-size 64 --reg 5e-4 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS}  --mining SemiHard \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-5 --feature-size 64 --reg 1e-4 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS} --mining KMeans \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 5e-7 --feature-size 64 --reg 1e-3 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS} --mining KMeans \
    --normalize-features
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 32 \
    --network ResNet --lr 1e-5 --feature-size 64 --reg 5e-4 --epochs 4 \
    --triplet-freq 1 --val-freq 1 --results-freq 1 --test-results-freq 1 \
    --batch-size 16  --loss ${LOSS}  --mining KMeans \
    --normalize-features
