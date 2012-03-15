NUMYLABELS=20

for C in 1
do
	for seed in 1
	do
		for foldnum in 0 1
		do
			for class in 'all'
			do
				for algorithm in 'CCCP' 'SPL'
				do
					for splControl in 0 1 
					do
						basedir=`./name.sh $C $seed $foldnum $class $algorithm $splControl`
						scriptname=jobs/${basedir}.sh

						CHANGE_DIR="cd $SPL_BASE_DIR"


						TRAIN="$EPYTHON train.py --seed=${seed} --splControl=${splControl} --splMode=$algorithm --C=${C} --scratchFile=output/$basedir.train --numYLabels=${NUMYLABELS} --dataFile=train/train.${class}_${foldnum}.txt --modelFile=output/${basedir}.model.cpz --latentVariableFile=output/${basedir}.train.latents >& output/${basedir}.train.output "
						TEST_ON_TRAIN="$EPYTHON test.py --scratchFile=output/$basedir.testOnTrain --dataFile=train/train.${class}_${foldnum}.txt --modelFile=output/${basedir}.model.cpz --numYLabels=${NUMYLABELS} --resultFile=output/${basedir}.train.results --latentVariableFile=${basedir}.testtrain.latents>& output/${basedir}.train.testoutput"

						TEST_ON_TEST="$EPYTHON test.py --scratchFile=output/$basedir.testOnTest --dataFile=train/test.${class}_${foldnum}.txt --modelFile=output/${basedir}.model.cpz --numYLabels=${NUMYLABELS} --resultFile=output/${basedir}.test.results --latentVariableFile=${basedir}.test.latents >& output/${basedir}.test.output"

						command_starttimestamp="date > ./output/${basedir}.starttime"
						command_hostname="hostname> ./output/${basedir}.hostname"
						command_endtimestamp="date > ./output/${basedir}.endtime" 
						command_starttime='START=$(date +%s)'
						command_endtime='END=$(date +%s)'
						command_difference='DIFF=$(( $END - $START ))'
						command_time_passed="echo \${DIFF} > ./output/${basedir}.totaltime"
						command_postprocess="./processOutput.sh ${basedir}"



						echo $CHANGE_DIR > $scriptname
						echo $command_starttimestamp >> $scriptname
						echo $command_hostname >> $scriptname
						echo $command_starttime >> $scriptname

						echo $TRAIN >> $scriptname
						echo $TEST_ON_TRAIN >> $scriptname
						echo $TEST_ON_TEST >> $scriptname
						echo $command_postprocess >> $scriptname

						echo $command_endtimestamp >> $scriptname
						echo $command_endtime >> $scriptname
						echo $command_difference >> $scriptname
						echo $command_time_passed >> $scriptname
						chmod +x $scriptname
						job=`pwd`/$scriptname

						if [ -z $1 ]
						then
							echo "Not posting $job" 
						else
							job=`pwd`/$scriptname
							echo "Posting $job"
							~/bin/appendJob.pl $job
	#						qsub -q daglab $scriptname
						fi
					done
				done
			done
		done
	done
done
