from imageImplementation import CommonApp
from imageImplementation import logging

import sys


def getScoresWLandY(w, example):
	return example.findScoreAllClasses(w), example.whiteList,example.trueY

def printStrongAndWeakTrainError(params, wBest):
	numCorrect = 0.0
	exampleScoresList = CommonApp.accessExamples(params,wBest,getScoresWLandY,  None)
	for scores,whiteList,trueY in exampleScoresList:
		bestLabel = -1
		maxScore = -1e100
		for l in range(params.numYLabels):
			if (scores[l] >= maxScore) and l not in whiteList:
				maxScore = scores[l]
				bestLabel = l

		assert(bestLabel >= 0)
		if bestLabel == trueY:
			numCorrect+=1
	trainingError = 1.0 - (numCorrect/float(params.numExamples))
	print("\nEvaluation error: %f"%(trainingError))

def getScoresandUUID(w,example):
	return example.findScoreAllClasses(w), example.fileUUID	

def writePerformance(params, w, resultFile):
	fh= open(resultFile, 'w')
	exampleScoresList = CommonApp.accessExamples(params,w,getScoresandUUID, None)  

	UUIDs = {}
	for exampleScores,UUID in exampleScoresList:
		if UUID in UUIDs:
			continue
		else:
			UUIDs[UUID] = 1
		fh.write("%s " %UUID)
		for key in exampleScores:
			fh.write("%d %f " %(key, exampleScores[key]))
		fh.write("\n")

	fh.close()
	
