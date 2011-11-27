import sys

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

	print("Weak training error: %f"%(1.0 - numWeaklyCorrect / float(params.numExamples)))
	print("Strong training error: %f"%(1.0 - numStronglyCorrect / float(params.numExamples)))

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
	

