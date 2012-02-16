cd /afs/cs.stanford.edu/u/rwitten/projects/splpython
date > ./output/SPL_C10_final_1_0.starttime
hostname> ./output/SPL_C10_final_1_0.hostname
START=$(date +%s)
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 train.py --initialModelFile=goodStartingPoints/CCCP.model.cpz --splControl=0 --splMode=SPL --C=10 --scratchFile=output/SPL_C10_final_1_0.train --dataFile=train/train.final_1.txt --modelFile=output/SPL_C10_final_1_0.model.cpz >& output/SPL_C10_final_1_0.train.output
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/SPL_C10_final_1_0.testOnTrain --dataFile=train/train.final_1.txt --modelFile=output/SPL_C10_final_1_0.model.cpz --numYLabels 20 --resultFile=output/SPL_C10_final_1_0.train.results >& output/SPL_C10_final_1_0.train.testoutput
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/SPL_C10_final_1_0.testOnTest --dataFile=train/test.final_1.txt --modelFile=output/SPL_C10_final_1_0.model.cpz --numYLabels 20 --resultFile=output/SPL_C10_final_1_0.test.results >& output/SPL_C10_final_1_0.test.output
date > ./output/SPL_C10_final_1_0.endtime
END=$(date +%s)
DIFF=$(( $END - $START ))
echo ${DIFF} > ./output/SPL_C10_final_1_0.totaltime
