import math
from math import Random

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
	elif splMode = 'SPL++':
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
	splMode = taskEachExample.parentTask.params.splParams.splMode
	ybar = taskEachExample.parentTask.curYbar
	k = taskEachExample.parentTask.curK
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
	splMode = taskEachTrueY.params.splParams.splMode
	numToSelect = ceil(taskEachtTrueY.globalSPLVars.fractionToSelect * float(len(sortedContribs) - retrieveNumWhiteListed(taskEachTrueY, splMode, ybar)))
	selectedSoFar = 0
	for i in range(len(sortedContribs)):
		if selectedSoFar < numToSelect:
			setSelected(sortedContribs[i][0], splMode, k, ybar, 1.0)
			selectedSoFar += 1
		elif ybar in sortedContribs[i][0].whiteList and taskEachTrueY.params.splParams.splMode == 'SPL++':
			setSelected(sortedContribs[i][0], splMode, k, ybar, 1.0)
		else:
			setSelected(sortedContribs[i][0], splMode, k, ybar, 0.0)

def selectKYbar(params, splMode):
	if splMode == 'SPL':
		return (None, None)
	elif splMode == 'SPL+':
		return (Random.randInt(0, params.numKernels - 1), None)
	elif splMode == 'SPL++':
		return (Random.randInt(0, params.numKernels - 1), Random.randInt(0, params.numYLabels - 1))
	else:
		print("ERROR: selectKYbar() doesn't support splMode = %s\n"%(splMode))
		assert(0)
		return (None, None)

def selectForEachTrueY(taskEachTrueY):
	numIters = taskEachTrueY.params.numSPLInnerIters
	for iter in range(numIters):
		(taskEachTrueY.curK, taskEachTrueY.curYbar) = selectKYbar(taskEachTrueY.params, taskEachTrueY.params.splParams.splMode)
		contributionsByExample = taskEachTrueY.processQueue.map(contributionForEachExample, taskEachTrueY.tasksByExample)
		selectLowestContributors(taskEachTrueY, contributionsByExample)

def getExamplesWithTrueY(params, trueY):
	trueYExamples = []
	for example in params.examples:
		if example.trueY == trueY:
			trueYExamples.append(example)

def initLocalSPLVars(taskEachTrueY, splMode):
	numK = 0
	numYbar = 0
	if splMode == 'SPL':
		numK = 1
		numYbar = 1
	elif splMode == 'SPL+':
		numK = taskEachTrueY.params.numKernels
		numYbar = 1
	elif splMode == 'SPL++':
		numK = taskEachTrueY.params.numKernels
		numYbar = taskEachTrueY.params.numYLabels
	else:
		assert(0)

	for k in range(numK):
		for ybar in range(numYbar):
			Random.shuffle(taskEachTrueY.tasksEachExample)
			taskList = []
			for i in range(len(taskEachTrueY.tasksEachExample)):
				taskList.append((taskEachTrueY.tasksEachExample[i], i))

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

def getPoolSize(params, maxNumAlive):
	return ceil(float(maxNumAlive) / float(params.numExamples) * 40.0)

def select(globalSPLVars, w, params):
	def jobifyEachTrueY(trueY):
		taskEachTrueY = SPLJob()
		def jobifyEachExample(example):
			taskEachExample = SPLJob()
			taskEachExample.parentTask = taskEachTrueY
			taskEachExample.localSPLVars = example.localSPLVars
			taskEachExample.fmvcJob = CommonApp.createFMVCJob(example, w, params)
			taskEachExample.whiteList = example.whiteList
			return taskEachExample

		taskEachTrueY.params = params
		taskEachTrueY.globalSPLVars = globalSPLVars
		trueYExamples = getExamplesWithTrueY(params, trueY)
		if params.splParams.splMode == 'SPL++':
			taskEachTrueY.numWhiteListed = getNumWhiteListed(params, trueYExamples)

		taskEachTrueY.processQueue = multiprocessing.Pool(getPoolSize(params, len(trueYExamples))) #TODO: figure out what it means to "create" a Pool with X number of worker processes.  Is that just the max number of processes that'll run simultaneously when I call Pool.map()?
		taskEachTrueY.trueY = trueY
		taskEachTrueY.tasksByExample = map(jobifyEachExample, trueYExamples) #hopefully won't have to parallelize this
		initLocalSPLVars(taskEachTrueY, params.splParams.splMode) #randomly include fraction of examples
		return taskEachTrueY

	tasksByTrueY = map(jobifyEachTrueY, range(params.numYLabels))
	params.processQueue.map(selectForEachTrueY, tasksByTrueY)
