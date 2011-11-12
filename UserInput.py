import getopt
import sys

from Params import Params

def setOptions(optdict):
	assert('--trainFile' in optdict) #this is the only required argument (for now)
	trainFile = optdict['--trainFile']
	assert(len(trainFile) > 0)
	params = Params()
	params.splParams = Params()
	params.epsilon = 0.01
	params.C = 1.0
	params.splParams.splMode = 'CCCP'
	params.seed = 0
	params.maxOuterIters = 20000
	params.syntheticParams = None
	params.numYLabels = 20
	kernelFile = 'dssdfsdfsfsdf'
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
		print("shoe")
		params.syntheticParams = Params()
		params.syntheticParams.numLatents = 10
		params.syntheticParams.strength = 3.0
		params.numExamples = 10
		params.lengthW = 2
		if '--syntheticNumLatents' in optdict:
			params.syntheticParams.numLatents = int(optdict['--syntheticNumLatents'])

		if '--syntheticNumExamples' in optdict:
			params.numExamples = int(optdict['--syntheticNumExamples'])

		if '--syntheticStrength' in optdict:
			params.syntheticParams.strength = float(optdict['--syntheticStrength'])

		if '--syntheticLengthW' in optdict:
			params.lengthW = int(optdict['--syntheticLengthW'])

	params.ylabels = range(params.numYLabels)
	params.maxDualityGap = params.C * params.epsilon
	return (params, trainFile, kernelFile)

def getUserInput():
	longOptions = ['trainFile=', 'numYLabels=', 'C=', 'epsilon=', 'splMode=', 'seed=', 'maxOuterIters=', 'kernelFile=', 'synthetic=', 'syntheticStrength=', 'syntheticNumExamples=', 'syntheticLengthW=', 'syntheticNumLatents=']
	(optlist, garbage) = getopt.getopt(sys.argv[1:], '', longOptions)
	optdict = {}
	for o, a in optlist:
		assert(o not in optdict) #we don't want people setting the same option twice
		optdict[o] = a

	(params, trainFile, kernelFile) = setOptions(optdict)
	return (params, trainFile, kernelFile)
