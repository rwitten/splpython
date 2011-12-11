import datetime
import math
import random
import numpy
import multiprocessing

from imageImplementation import CommonApp

class SPLJob(object):
	def __init__(self):
		pass

class SPLVar(object):
	def __init__(self):
		pass

def setSelected(taskEachExample, splMode, k, ybar, value):
	if splMode == 'SPL':
		taskEachExample.localSPLVars.selected = value
	elif splMode == 'SPL+':
		taskEachExample.localSPLVars.selected[0, k] = value
	elif splMode == 'SPL++':
		taskEachExample.localSPLVars.selected[ybar][0, k] = value
	else:
		print("ERROR: setSelected() does not support splMode = %s\n"%(splMode))
		assert(0)

def setupSPL(examples, params):
	def setupSPLEachExample(example):
		example.localSPLVars = SPLVar()
		if params.splParams.splMode == 'SPL':
			example.localSPLVars.selected = 1.0
		elif params.splParams.splMode == 'SPL+':
			example.localSPLVars.selected = numpy.ones((1, params.numKernels))
		elif params.splParams.splMode == 'SPL++':
			example.localSPLVars.selected = []
			for i in range(params.numYLabels):
				example.localSPLVars.selected.append(numpy.ones((1, params.numKernels)))
		else:
			assert(0)

	map(setupSPLEachExample, examples)

#note: This must return a (taskEachExample, contribution) tuple
def contributionForEachExample(taskEachExample):
	k = taskEachExample.curK
	ybar = taskEachExample.curYbar
	splMode = taskEachExample.splMode
	setSelected(taskEachExample, splMode, k, ybar, 1.0)
	(garbageA, garbageB, slackOn) = CommonApp.singleFMVC(taskEachExample.fmvcJob)
	setSelected(taskEachExample, splMode, k, ybar, 0.0)
	(garbageA, garbageB, slackOff) = CommonApp.singleFMVC(taskEachExample.fmvcJob)
	contribTuple = (taskEachExample, (slackOn - slackOff))
	return contribTuple

def retrieveNumWhiteListed(taskEachTrueY, splMode, ybar):
	if splMode == 'SPL++':
		return taskEachTrueY.numWhiteListed[ybar]
	else:
		return 0

def selectLowestContributors(taskEachTrueY, contributionsByExample):
	sortedContribs = sorted(contributionsByExample, key=lambda contrib: contrib[1])
	ybar = taskEachTrueY.curYbar
	k = taskEachTrueY.curK
	splMode = taskEachTrueY.splMode
	numToSelect = math.ceil(taskEachTrueY.fraction * float(len(sortedContribs) - retrieveNumWhiteListed(taskEachTrueY, splMode, ybar)))
	selectedSoFar = 0
	for i in range(len(sortedContribs)):
		if selectedSoFar < numToSelect:
			setSelected(sortedContribs[i][0], splMode, k, ybar, 1.0)
			selectedSoFar += 1
		elif ybar in sortedContribs[i][0].whiteList and taskEachTrueY.splMode == 'SPL++':
			setSelected(sortedContribs[i][0], splMode, k, ybar, 1.0)
		else:
			setSelected(sortedContribs[i][0], splMode, k, ybar, 0.0)

def getKYList(numKernels, numYLabels, splMode):
	kyList = []
	if splMode == 'SPL':
		kyList.append((None, None))
	elif splMode == 'SPL+':
		for k in range(numKernels):
			kyList.append((k, None))

	elif splMode == 'SPL++':
		for k in range(numKernels):
			for y in range(numYLabels):
				kyList.append((k, y))

	else:
		assert(0)

	random.shuffle(kyList)
	return kyList

def selectForEachTrueY(taskEachTrueY):
	def setKYbar(taskEachExample):
		taskEachExample.curK = taskEachTrueY.curK
		taskEachExample.curYbar = taskEachTrueY.curYbar

	if len(taskEachTrueY.tasksByExample) <= 0: #Based on what I've seen so far, I don't trust Pool.map() to deal with edge cases like this
		return

	curProcess = multiprocessing.current_process()
	curProcess.daemon = False #daemonic processes can't have children
	poolSize = max(getPoolSize(taskEachTrueY.totalNumExamples, len(taskEachTrueY.tasksByExample)), 1)
	processQueue = multiprocessing.Pool(poolSize)
	numIters = taskEachTrueY.splInnerIters
	for iter in range(numIters):
		kyList = getKYList(taskEachTrueY.numKernels, taskEachTrueY.numYLabels, taskEachTrueY.splMode)
		for kyPair in kyList:
			(taskEachTrueY.curK, taskEachTrueY.curYbar) = (kyPair[0], kyPair[1])
			map(setKYbar, taskEachTrueY.tasksByExample)
			contributionsByExample = processQueue.map(contributionForEachExample, taskEachTrueY.tasksByExample)
			#contributionsByExample = map(contributionForEachExample, taskEachTrueY.tasksByExample)
			selectLowestContributors(taskEachTrueY, contributionsByExample)


	deathstart = datetime.datetime.now()
	processQueue.close()
	processQueue.join()
	deathend = datetime.datetime.now()
	#print("destroying pool takes %f seconds\n"%((deathend-deathstart).total_seconds()))
	#print("finished!\n")

def getExamplesWithTrueY(params, trueY):
	trueYExamples = []
	for example in params.examples:
		if example.trueY == trueY:
			trueYExamples.append(example)

	return trueYExamples

def initLocalSPLVars(taskEachTrueY, splMode):
	numK = 0
	numYbar = 0
	if splMode == 'SPL':
		numK = 1
		numYbar = 1
	elif splMode == 'SPL+':
		numK = taskEachTrueY.numKernels
		numYbar = 1
	elif splMode == 'SPL++':
		numK = taskEachTrueY.numKernels
		numYbar = taskEachTrueY.numYLabels
	else:
		assert(0)

	for k in range(numK):
		for ybar in range(numYbar):
			random.shuffle(taskEachTrueY.tasksByExample)
			taskList = []
			for i in range(len(taskEachTrueY.tasksByExample)):
				taskList.append((taskEachTrueY.tasksByExample[i], i))

			taskEachTrueY.curK = k
			taskEachTrueY.curYbar = ybar
			selectLowestContributors(taskEachTrueY, taskList)

def getNumWhiteListed(params, trueYExamples):
	numWhiteListed = []
	for ybar in range(params.numYLabels):
		num = 0
		for example in trueYExamples:
			if ybar in example.whiteList:
				num += 1

		numWhiteListed.append(num)

	return numWhiteListed

def getPoolSize(totalNumExamples, maxNumAlive):
	return int(math.ceil(float(maxNumAlive) / float(totalNumExamples) * 40.0))

def select(globalSPLVars, w, params):
	def jobifyEachTrueY(trueY):
		taskEachTrueY = SPLJob()
		def jobifyEachExample(example):
			taskEachExample = SPLJob()
			taskEachExample.splMode = taskEachTrueY.splMode
			taskEachExample.localSPLVars = example.localSPLVars
			taskEachExample.fmvcJob = CommonApp.createFMVCJob(example, params,w)
			taskEachExample.whiteList = example.whiteList
			return taskEachExample

		taskEachTrueY.totalNumExamples = params.numExamples
		taskEachTrueY.numYLabels = params.numYLabels
		taskEachTrueY.numKernels = params.numKernels
		taskEachTrueY.splMode = params.splParams.splMode
		taskEachTrueY.splInnerIters = params.splParams.splInnerIters
		taskEachTrueY.fraction = globalSPLVars.fraction
		trueYExamples = getExamplesWithTrueY(params, trueY)
		if params.splParams.splMode == 'SPL++':
			taskEachTrueY.numWhiteListed = getNumWhiteListed(params, trueYExamples)

		taskEachTrueY.trueY = trueY
		taskEachTrueY.tasksByExample = map(jobifyEachExample, trueYExamples) #hopefully won't have to parallelize this
		initLocalSPLVars(taskEachTrueY, params.splParams.splMode) #randomly include fraction of examples
		return taskEachTrueY

	splstart = datetime.datetime.now() 
	tasksByTrueY = map(jobifyEachTrueY, range(params.numYLabels))
	params.processPool.map(selectForEachTrueY, tasksByTrueY)
	#map(selectForEachTrueY, tasksByTrueY)
	splend = datetime.datetime.now()
	print("SPL update took %f seconds\n"%((splend - splstart).total_seconds()))
	#print("by the time I get printed, everything should be finished\n")
	#map(selectForEachTrueY, tasksByTrueY)
