cd /afs/cs.stanford.edu/u/rwitten/projects/splpython
date > ./output/SPL_C1000_final_1_1.starttime
hostname> ./output/SPL_C1000_final_1_1.hostname
START=$(date +%s)
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 train.py --initialModelFile=goodStartingPoints/CCCP.model.cpz --splControl=1 --splMode=SPL --C=1000 --scratchFile=output/SPL_C1000_final_1_1.train --dataFile=train/train.final_1.txt --modelFile=output/SPL_C1000_final_1_1.model.cpz >& output/SPL_C1000_final_1_1.train.output
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/SPL_C1000_final_1_1.testOnTrain --dataFile=train/train.final_1.txt --modelFile=output/SPL_C1000_final_1_1.model.cpz --numYLabels 20 --resultFile=output/SPL_C1000_final_1_1.train.results >& output/SPL_C1000_final_1_1.train.testoutput
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/SPL_C1000_final_1_1.testOnTest --dataFile=train/test.final_1.txt --modelFile=output/SPL_C1000_final_1_1.model.cpz --numYLabels 20 --resultFile=output/SPL_C1000_final_1_1.test.results >& output/SPL_C1000_final_1_1.test.output
date > ./output/SPL_C1000_final_1_1.endtime
END=$(date +%s)
DIFF=$(( $END - $START ))
echo ${DIFF} > ./output/SPL_C1000_final_1_1.totaltime
