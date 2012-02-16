cd /afs/cs.stanford.edu/u/rwitten/projects/splpython
date > ./output/NAGSCCCP_C10_all_1.starttime
hostname> ./output/NAGSCCCP_C10_all_1.hostname
START=$(date +%s)
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 train.py --splMode=CCCP --C=10 --scratchFile=output/NAGSCCCP_C10_all_1.train --dataFile=train/train.all_1.txt --modelFile=output/NAGSCCCP_C10_all_1.model.cpz >& output/NAGSCCCP_C10_all_1.train.output
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/NAGSCCCP_C10_all_1.testOnTrain --dataFile=train/train.all_1.txt --modelFile=output/NAGSCCCP_C10_all_1.model.cpz --numYLabels 20 --resultFile=output/NAGSCCCP_C10_all_1.train.results >& output/NAGSCCCP_C10_all_1.train.testoutput
/afs/cs.stanford.edu/u/rwitten/libs/epd/bin/python2.7 test.py --scratchFile=output/NAGSCCCP_C10_all_1.testOnTest --dataFile=train/test.all_1.txt --modelFile=output/NAGSCCCP_C10_all_1.model.cpz --numYLabels 20 --resultFile=output/NAGSCCCP_C10_all_1.test.results >& output/NAGSCCCP_C10_all_1.test.output
date > ./output/NAGSCCCP_C10_all_1.endtime
END=$(date +%s)
DIFF=$(( $END - $START ))
echo ${DIFF} > ./output/NAGSCCCP_C10_all_1.totaltime
