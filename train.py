from imageImplementation import ImageApp as App

import LSSVM
import PsiCache
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
		params.examples.append(App.ImageExample(line, params, len(params.examples)))
	
	print "total number of examples (including duplicates) = " + repr(params.numExamples)
	sys.stdout.write('\n')
	assert(params.numExamples == len(params.examples))
	params.maxDualityGap = params.C*params.epsilon

def main():
	(params, trainFile, kernelFile) = UserInput.getUserInput()

	App.loadKernelFile(kernelFile, params) 
	loadTrainFile(trainFile, params)

	w = App.PsiObject(params,False)
	LSSVM.optimize(w, params)
 
	return params

if __name__== "__main__":
	main()
