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

def main():
	(params, trainFile, kernelFile) = UserInput.getUserInput()

	if params.syntheticParams:
		synthesizeExamples(params)
	else:
		App.loadKernelFile(kernelFile, params)
		loadTrainFile(trainFile, params)

	w = App.PsiObject(params,False)
	LSSVM.optimize(w, params)
 
	return params

if __name__== "__main__":
	main()
