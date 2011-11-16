import getopt
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
	params.splParams = Params()
	params.epsilon = .01
	params.C = 1.0
	params.splParams.splMode = 'CCCP'
	params.seed = 0
	params.maxOuterIters = 20000
	params.syntheticParams = None
	params.supervised = False
	params.numYLabels = 20
	kernelFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/allkernels_info.txt'
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
		params.syntheticParams.strength = 3.0
		params.numExamples = 10
		params.totalLength = params.numYLabels + 1
		params.lengthW = params.numYLabels * params.totalLength
		if '--syntheticNumLatents' in optdict:
			params.syntheticParams.numLatents = int(optdict['--syntheticNumLatents'])

		if '--syntheticNumExamples' in optdict:
			params.numExamples = int(optdict['--syntheticNumExamples'])

		if '--syntheticStrength' in optdict:
			params.syntheticParams.strength = float(optdict['--syntheticStrength'])

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
		return params

def getUserInput(train_or_test):
	longOptions = ['modelFile=', 'dataFile=', 'numYLabels=', 'C=', 'epsilon=', 'splMode=', 'seed=', 'maxOuterIters=', 'kernelFile=', 'synthetic=', 'syntheticStrength=', 'syntheticNumExamples=', 'syntheticNumLatents=', 'supervised=']
	if train_or_test == 'test':
		longOptions.append('resultFile=')

	(optlist, garbage) = getopt.getopt(sys.argv[1:], '', longOptions)
	optdict = {}
	for o, a in optlist:
		assert(o not in optdict) #we don't want people setting the same option twice
		optdict[o] = a

	arg_list = setOptions(optdict, train_or_test)
	return arg_list
