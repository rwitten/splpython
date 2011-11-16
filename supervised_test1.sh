#!/bin/bash
cd scratch/autumn2011/splpython
epython train.py --dataFile='train/train.all_1.txt' --modelFile='output/newsmall_model1.cpz' --supervised=1 >& supervised_test1_output
epython test.py --dataFile='train/train.all_1.txt' --modelFile='output/newsmall_model1.cpz' --supervised=1 --numYLabels 2 --resultFile='output/results'  
