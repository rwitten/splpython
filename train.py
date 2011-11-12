from imageImplementation import ImageApp as App

import LSSVM
import PsiCache

import sys
import scipy

class Params(object):
    pass

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
	params.maxDualityGap = params.C*params.epsilon

def main():
	kernelFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/allkernels_info.txt'
	#trainFile = 'train/train.1.txt'
	trainFile = 'train/train.newsmall_1_reducedy.txt'

	params = Params()
	params.splParams = Params()
	params.splParams.splMode = 'CCCP'
	params.max_outer_iter = 2#TODO: make this user input
	App.loadKernelFile(kernelFile, params) 
	loadTrainFile(trainFile, params)

	w = App.PsiObject(params,False)
	LSSVM.optimize(w, params)
 
	return params

if __name__== "__main__":
	main()
