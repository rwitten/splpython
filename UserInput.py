import getopt
import sys

import Params

def setOptions(optdict):
	trainFile = optdict['--trainFile']
	assert(len(trainFile) > 0)
	params = Params()
	params.splParams = Params()
	params.epsilon = 0.01
	params.C = 1.0
	params.splParams.splMode = 'CCCP'
	params.seed = 0
	params.maxOuterIters = 20000
	params.synthetic = 0
	params.numClasses = 20
	kernelFile = 'dssdfsdfsfsdf'
	if optdict['--kernelFile']:
		kernelFile = optdict['--kernelFile']

	if optdict['--C']:
		params.C = float(optdict['--C'])
	
	if optdict['--epsilon']:
		params.epsilon = float(optdict['--epsilon'])
	
	if optdict['--splMode']:
		params.splParams.splMode = optdict['--splMode']
	
	if optdict['--seed']:
		params.seed = int(optdict['--seed'])
	
	if optdict['--maxOuterIters']:
		params.maxOuterIters = int(optdict['--maxOuterIters'])
	
	if optdict['--numClasses']:
		params.numClasses = int(optdict['--numClasses'])

	if optdict['--synthetic'i]:
		params.synthetic = bool(optdict['--synthetic'])
	
	return (params, trainFile, kernelFile)

def getUserInput():
	print "hi"
	longOptions = ['trainFile=', 'numClasses', 'C', 'epsilon', 'splMode', 'seed', 'maxOuterIters', 'kernelFile', 'synthetic']
	(optlist, garbage) = getopt.getopt(sys.argv[1:], longOptions)
	optdict = {}
	for o, a in optlist:
		assert(o not in optdict) #we don't want people setting the same option twice
		optdict[o] = a

	(params, trainFile, kernelFile) = setOptions(optdict)
	return (params, trainFile, kernelFile)
