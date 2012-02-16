cd /afs/cs.stanford.edu/u/rwitten/projects/splpython
date > ./output/CCCP_C10_final_1_1.starttime
hostname> ./output/CCCP_C10_final_1_1.hostname
START=$(date +%s)
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 train.py --initialModelFile=goodStartingPoints/CCCP.model.cpz --splControl=1 --splMode=CCCP --C=10 --scratchFile=output/CCCP_C10_final_1_1.train --dataFile=train/train.final_1.txt --modelFile=output/CCCP_C10_final_1_1.model.cpz >& output/CCCP_C10_final_1_1.train.output
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/CCCP_C10_final_1_1.testOnTrain --dataFile=train/train.final_1.txt --modelFile=output/CCCP_C10_final_1_1.model.cpz --numYLabels 20 --resultFile=output/CCCP_C10_final_1_1.train.results >& output/CCCP_C10_final_1_1.train.testoutput
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/CCCP_C10_final_1_1.testOnTest --dataFile=train/test.final_1.txt --modelFile=output/CCCP_C10_final_1_1.model.cpz --numYLabels 20 --resultFile=output/CCCP_C10_final_1_1.test.results >& output/CCCP_C10_final_1_1.test.output
date > ./output/CCCP_C10_final_1_1.endtime
END=$(date +%s)
DIFF=$(( $END - $START ))
echo ${DIFF} > ./output/CCCP_C10_final_1_1.totaltime
