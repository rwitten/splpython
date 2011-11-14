from imageImplementation import ImageApp as App

import LSSVM
from imageImplementation import PsiCache
import UserInput

import sys
import scipy

def loadTrainFile(trainFile, params):
	tFile = open(trainFile,'r')
	params.numExamples= int(tFile.readline())
	params.examples = []
	params.cache= PsiCache.PsiCache()
	for line in tFile:
		sys.stdout.write("%")
		sys.stdout.flush()
		params.examples.append(App.ImageExample(line, params, len(params.examples)))
	
	print "total number of examples (including duplicates) = " + repr(params.numExamples)
	sys.stdout.write('\n')
	assert(params.numExamples == len(params.examples))

def synthesizeExamples(params):
	params.cache = PsiCache.PsiCache()
	params.examples = []
	for i in range(params.numExamples):
		params.examples.append(App.ImageExample(None, params, i))

def writeModel(params, modelFile, w):
	mf = open(modelFile, "w")
	mf.write("%d "%(params.lengthW))
	for i in range(lengthW - 1):
		mf.write("%lf "%(w[i, 0]))

	mf.write("%lf"%(w[params.lengthW - 1, 0]))
	mf.close()

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

def main():
	(params, trainFile, kernelFile, modelFile) = UserInput.getUserInput('train')

	if params.syntheticParams:
		synthesizeExamples(params)
	else:
		App.loadKernelFile(kernelFile, params)
		loadTrainFile(trainFile, params)

	w = App.PsiObject(params,False)
	wBest = LSSVM.optimize(w, params)

	printStrongAndWeakTrainError(params, wBest) #yeah, I know, I'm not supposed to do this here, but this is really just a sanity check
 
	writeModel(params, modelFile, wBest)

	return params

if __name__== "__main__":
	main()
