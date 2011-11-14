#!/bin/bash
cd scratch/autumn2011/splpython
python train.py --dataFile='/afs/cs/u/rwitten/projects/multi_kernel_spl/data/train.newsmall_1.txt' --modelFile='newsmall_model1.txt' --supervised=1 >& supervised_test1_output
