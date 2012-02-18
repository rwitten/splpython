#!/bin/bash

basename=$1

./ap_compute.sh output/${basename}.train.results output/${basename}.train.ap
./ap_compute.sh output/${basename}.test.results output/${basename}.test.ap

cat output/${basename}.train.output | grep "Best objective attained" | awk '{print $5;}' > output/${basename}.objective

cat output/${basename}.test.output | grep "Evaluation error" | awk '{print $3;}' > output/${basename}.test.loss
cat output/${basename}.train.testoutput | grep "Evaluation error" | awk '{print $3;}' > output/${basename}.train.loss
