from imageImplementation import CacheObj
from imageImplementation import CommonApp
from imageImplementation import ImageApp as App
from imageImplementation import PsiCache

import logging
import multiprocessing
import sys

def loadDataFile(trainFile, params):
	tFile = open(trainFile,'r')
	params.numExamples= int(tFile.readline())
	params.examples = []
	params.cache= PsiCache.PsiCache()

	for line in tFile:
		params.examples.append(App.ImageExample(line, params, len(params.examples)))
	
	logging.debug("total number of examples (including duplicates) = " + repr(params.numExamples))
	assert(params.numExamples == len(params.examples))

def synthesizeExamples(params):
	params.cache = PsiCache.PsiCache()
	params.examples = []
	for i in range(params.numExamples):
		params.examples.append(App.ImageExample(None, params, i))


def printStrongAndWeakTrainError(params, wBest):
	numStronglyCorrect = 0.0
	numWeaklyCorrect = 0.0
	for i in range(params.numExamples):
		scores = params.examples[i].findScoreAllClasses(wBest)


		bestLabel = -1
		maxScore = -1e100
		for l in range(params.numYLabels):
			if scores[l] >= maxScore:
				maxScore = scores[l]
				bestLabel = l

		assert(bestLabel >= 0)
		stronglyCorrect = 0.0
		weaklyCorrect = 0.0
		if bestLabel == params.examples[i].trueY:
			weaklyCorrect = 1.0
			stronglyCorrect = 1.0

		if bestLabel in params.examples[i].whiteList:
			weaklyCorrect = 1.0

		numStronglyCorrect += stronglyCorrect
		numWeaklyCorrect += weaklyCorrect

	logging.debug("Weak training error: %f"%(1.0 - numWeaklyCorrect / float(params.numExamples)))
	logging.debug("Strong training error: %f"%(1.0 - numStronglyCorrect / float(params.numExamples)))

def writePerformance(params, w, resultFile):
	fh= open(resultFile, 'w')
	UUIDs = set()

	for example in params.examples:
		if example.fileUUID in UUIDs:
			continue
		else:
			UUIDs.add(example.fileUUID)
		
		exampleScores= example.findScoreAllClasses(w)
		fh.write("%s " %example.fileUUID)
		for key in exampleScores:
			fh.write("%d %f " %(key, exampleScores[key]))
		fh.write("\n")

	fh.close()

def grabLatentVariables(emptyBlob, example):
	return example.fileUUID, example.hlabels[example.h]

def dumpCurrentLatentVariables(params, lvFile):
	idLVpairs = CommonApp.accessExamples(params, None, grabLatentVariables, None)
	fh = open(lvFile,'w')
	idLVpairs.sort( lambda x,y: int(x[0])-int(y[0]))

	for id,LV in idLVpairs:
		fh.write("%s %d %d %d %d\n" % (id, LV.x_min, LV.x_max, LV.y_min, LV.y_max))

	fh.close()
