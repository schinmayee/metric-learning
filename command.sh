#!/usr/bin/env bash
./train.py --num-train 80 --num-val 40 --num-test 80 --triplets-per-class 64 --network Squeeze --lr 1e-5 --feature-size 128
