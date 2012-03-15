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

def setupSPLEachExample(ignore, example):
	example.localSPLVars = SPLVar()
	if example.params.splParams.splMode == 'SPL':
		example.localSPLVars.selected = 0.0
#	elif params.splParams.splMode == 'SPL+':
#		example.localSPLVars.selected = numpy.ones((1, params.numKernels))
#	elif params.splParams.splMode == 'SPL++':
#		example.localSPLVars.selected = []
#		for i in range(params.numYLabels):
#			example.localSPLVars.selected.append(numpy.ones((1, params.numKernels)))
	else:
		assert(0)

def setupSPL(params):
	CommonApp.accessExamples(params, None, setupSPLEachExample, None)

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


def findCutoffs(allViolations, labels, numYLabels,fraction):
	cutoffDict= {}
	for index in range(numYLabels):
		relevantExamples = filter( lambda violationLabelTuple : violationLabelTuple[1]==index ,zip(allViolations, labels))
		violations = map(lambda violationLabelTuple: violationLabelTuple[0], relevantExamples)
		violations.sort()
		print("len(violations) %f fraction %f count %f index %d" %( len(violations), fraction, int(round(len(violations)*fraction+.01))-1,index ))
		assert(len(violations)>0)
		cutoffDict[index] = violations[int(round(len(violations)*fraction+.01))-1]

	return cutoffDict

def computeViolation(scoreDict, trueY):
	maxOpposingScore = -1e10
	for key in scoreDict:
		if key != trueY:
			if maxOpposingScore < scoreDict[key]:
				maxOpposingScore= scoreDict[key]
	
	return scoreDict[trueY] - maxOpposingScore
	
def findYAndViolation( w, example):
	return example.trueY, computeViolation(example.findScoreAllClasses(w, True), example.trueY)

def updateSelectionSPL(wCutoffDictandFraction, example):
	w= wCutoffDictandFraction[0]
	cutoffDict= wCutoffDictandFraction[1]
	fraction = wCutoffDictandFraction[2]
	ignore, violation = findYAndViolation(w, example)
	if not example.params.splParams.splControl:
		if cutoffDict[example.trueY] >= violation:
			example.localSPLVars.selected = 1.0
		else:
			example.localSPLVars.selected = 0
	else:
		temp = random.random()
		if example.params.splParams.splControl == 1:
			if temp<fraction:
				example.localSPLVars.selected = 1.0
			else:
				example.localSPLVars.selected = 0
		else:
			assert(example.params.splParams.splControl == 2)
			oldfraction = fraction - example.params.splParams.splIncrement
			incrementfraction = example.params.splParams.splIncrement/(1-oldfraction)
			if example.localSPLVars.selected or temp<incrementfraction:
				example.localSPLVars.selected = 1.0
			else:
				example.localSPLVars.selected = 0

	return (example.trueY, example.localSPLVars.selected)

def getTrueY(example):
	return example.trueY


def select(globalSPLVars, w, params):
	print("Fraction %f" % globalSPLVars.fraction)
	labels, violations= zip(*CommonApp.accessExamples(params,w, findYAndViolation, None))
	cutoffDict = findCutoffs(violations, labels, params.numYLabels,globalSPLVars.fraction)
	trueYAndSelections = CommonApp.accessExamples(params, (w, cutoffDict,globalSPLVars.fraction), updateSelectionSPL, None)	

	for label in range(params.numYLabels):
		total = 0
		totalOn = 0
		for trueYandSelection in trueYAndSelections:
			if trueYandSelection[0]==label:
				totalOn += trueYandSelection[1]
				total+=1
		print("For label %d totalOn %f and total %f" %( label, totalOn, total))

