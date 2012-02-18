import getopt
import multiprocessing
#from multiprocessing import dummy as multiprocessing
import sys

from Params import Params


def setOptions(optdict, train_or_test):
	assert('--dataFile' in optdict)
	dataFile = optdict['--dataFile']
	assert(len(dataFile) > 0)
	assert('--modelFile' in optdict)
	modelFile = optdict['--modelFile']
	assert(len(modelFile) > 0)
	
	params = Params()
	params.trainOrTest = train_or_test 
	params.splParams = Params()
	params.epsilon = .01
	params.C = 1.0
	params.splParams.splMode = 'CCCP'
	params.seed = 0
	params.maxOuterIters = 20000
	params.estimatedNumConstraints = 100
	params.syntheticParams = None
	params.supervised = False
	params.numYLabels = 20
	params.maxPsiGap = 0.00001
	params.maxIdleIters = 10
	params.splParams.splInitFraction = 0.5
	params.splParams.splIncrement = 0.1
	params.splParams.splInitIters = 0
	params.splParams.splOuterIters = 1
	params.splParams.splInnerIters = 1

	params.babyData = 0
	params.balanceClasses = 0

#	assert('--splControl' in optdict)
	if '--splControl' in optdict:
		params.splParams.splControl = int(optdict['--splControl'])

	if '--splInitIters' in optdict:
		params.splParams.splInitIters = int(optdict['--splInitIters'])

	if '--balanceClasses' in optdict:
		params.balanceClasses = int(optdict['--balanceClasses'])

	if '--babyData' in optdict:
		params.babyData = int(optdict['--babyData'])

	if params.babyData == 1:
		params.numYLabels = 7

	kernelFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/allkernels_info.txt'
	if '--maxPsiGap' in optdict:
		params.maxPsiGap = float(optdict['--maxPsiGap'])
	
	params.initialModelFile = None
	if '--initialModelFile' in optdict:
		params.initialModelFile = optdict['--initialModelFile']

	if '--maxTimeIdle' in optdict:
		params.maxTimeIdle = int(optdict['--maxTimeIdle'])

	if '--supervised' in optdict:
		params.supervised = optdict['--supervised']

	if '--kernelFile' in optdict:
		kernelFile = optdict['--kernelFile']

	if '--C' in optdict:
		params.C = float(optdict['--C'])
	
	if '--epsilon' in optdict:
		params.epsilon = float(optdict['--epsilon'])
	
	if '--splMode' in optdict:
		params.splParams.splMode = optdict['--splMode']
	
	if '--seed' in optdict:
		params.seed = int(optdict['--seed'])
	
	if '--maxOuterIters' in optdict:
		params.maxOuterIters = int(optdict['--maxOuterIters'])
	
	if '--numYLabels' in optdict:
		params.numYLabels = int(optdict['--numYLabels'])

	if '--synthetic' in optdict and int(optdict['--synthetic']):
		params.syntheticParams = Params()
		params.syntheticParams.numLatents = 10
		params.syntheticParams.strength = 10.0
		params.numYLabels = 5
		params.maxPsiGap = 0.00001
		params.numExamples = 50
		params.totalLength = params.numYLabels + 1
		params.lengthW = params.numYLabels * params.totalLength
		if '--syntheticNumLatents' in optdict:
			params.syntheticParams.numLatents = int(optdict['--syntheticNumLatents'])

		if '--syntheticNumExamples' in optdict:
			params.numExamples = int(optdict['--syntheticNumExamples'])

		if '--syntheticStrength' in optdict:
			params.syntheticParams.strength = float(optdict['--syntheticStrength'])

	assert('--scratchFile' in optdict)
	params.scratchFile = optdict['--scratchFile']

	params.ylabels = range(params.numYLabels)
	params.maxDualityGap = params.C * params.epsilon
	params.dataFile= dataFile
	params.kernelFile = kernelFile
	params.modelFile = modelFile
	if train_or_test == 'test':
		assert('--resultFile' in optdict)
		resultFile = optdict['--resultFile']
		params.resultFile = resultFile
		assert(len(resultFile) > 0)
		return params
	else:
		assert(train_or_test =='train')
		numConsumers = multiprocessing.cpu_count()
#		params.inputQueues = [Queue() for i in range(numConsumers)]
#		params.outputQueue= Queue()
#		params.processes = [ConsumerProcess(str(i),i , inputQueues[i], outputQueue) for i in range(numConsumers)]
#		for i in range(numConsumers):
#			params.processes[p].start()

		#params.processPool = multiprocessing.Pool()
		return params


def getUserInput(train_or_test):
	longOptions = ['modelFile=', 'dataFile=', 'numYLabels=', 'C=', 'epsilon=', 'splMode=', 'seed=', 'maxOuterIters=', 'kernelFile=', 'synthetic=', 'syntheticStrength=', 'syntheticNumExamples=', 'syntheticNumLatents=', 'supervised=', 'maxPsiGap=', 'maxTimeIdle=', 'scratchFile=', 'babyData=', 'balanceClasses=', 'initialModelFile=', 'splInitIters=', 'splControl=']


	if train_or_test == 'test':
		longOptions.append('resultFile=')

	(optlist, garbage) = getopt.getopt(sys.argv[1:], '', longOptions)
	optdict = {}
	for o, a in optlist:
		assert(o not in optdict) #we don't want people setting the same option twice
		optdict[o] = a

	arg_list = setOptions(optdict, train_or_test)
	return arg_list
