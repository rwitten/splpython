for C in 1 10 100
do
	for foldnum in 1
	do
		for class in 'all'
		do
			basedir=`./name.sh $C $foldnum $class`
			scriptname=jobs/${basedir}.sh

			CHANGE_DIR="cd $SPL_BASE_DIR"


			TRAIN="$EPYTHON train.py --C=${C} --scratchFile=output/$basedir.train --dataFile=train/train.${class}_${foldnum}.txt --modelFile=output/${basedir}.model.cpz >& output/${basedir}.train.output"
			TEST_ON_TRAIN="$EPYTHON test.py --scratchFile=output/$basedir.testOnTrain --dataFile=train/train.${class}_${foldnum}.txt --modelFile=output/${basedir}.model.cpz --numYLabels 20 --resultFile=output/${basedir}.train.results >& output/${basedir}.train.testoutput"

			TEST_ON_TEST="$EPYTHON test.py --scratchFile=output/$basedir.testOnTest --dataFile=train/test.${class}_${foldnum}.txt --modelFile=output/${basedir}.model.cpz --numYLabels 20 --resultFile=output/${basedir}.test.results >& output/${basedir}.test.output"

			command_starttimestamp="date > ./output/${basedir}.starttime"
			command_hostname="hostname> ./output/${basedir}.hostname"
			command_endtimestamp="date > ./output/${basedir}.endtime" 
			command_starttime='START=$(date +%s)'
			command_endtime='END=$(date +%s)'
			command_difference='DIFF=$(( $END - $START ))'
			command_time_passed="echo \${DIFF} > ./output/${basedir}.totaltime"


			echo $CHANGE_DIR > $scriptname
			echo $command_starttimestamp >> $scriptname
			echo $command_hostname >> $scriptname
			echo $command_starttime >> $scriptname

			echo $TRAIN >> $scriptname
			echo $TEST_ON_TRAIN >> $scriptname
			echo $TEST_ON_TEST >> $scriptname

			echo $command_endtimestamp >> $scriptname
			echo $command_endtime >> $scriptname
			echo $command_difference >> $scriptname
			echo $command_time_passed >> $scriptname
			chmod +x $scriptname

			if [ -z $1 ]
			then
				echo "Not posting $scriptname" 
			else
				echo "Posting $scriptname"
				qsub -q daglab $scriptname
			fi

		done
	done
done
