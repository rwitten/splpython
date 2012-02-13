import sys

def printStrongAndWeakTrainError(params, wBest):
	numCorrect = 0.0
	for i in range(params.numExamples):
		scores = params.examples[i].findScoreAllClasses(wBest)

		bestLabel = -1
		maxScore = -1e100
		for l in range(params.numYLabels):
			if (scores[l] >= maxScore) and l not in params.examples[i].whiteList:
				maxScore = scores[l]
				bestLabel = l

		assert(bestLabel >= 0)
		if bestLabel == params.examples[i].trueY:
			numCorrect+=1

	trainingError = 1.0 - (numCorrect/float(params.numExamples))
	print("Evaluation error: %f"%(trainingError))

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
	

