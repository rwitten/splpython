from imageImplementation import ImageApp as App
import LSSVM

import sys
import scipy

class Params(object):
    pass

def loadTrainFile(trainFile, params):
	tFile = open(trainFile,'r')
	params.numUniqueExamples = int(tFile.readline())
	params.examples = []
	for line in tFile:
		sys.stdout.write("%")
		params.examples.append(App.ImageExample(line, params, len(params.examples), None, -1))
	
	for i in range(params.numUniqueExamples):
		for j in range(len(params.examples[i].whiteList)):
			params.examples.append(App.ImageExample(None, params, len(params.examples), params.examples[i], j))

	params.numExamples = len(params.examples)
	print "total number of examples (including duplicates) = " + repr(params.numExamples)
	sys.stdout.write('\n')
	assert(params.numExamples == len(params.examples))
	params.maxDualityGap = params.C*params.epsilon #*params.numExamples #TODO: I took away params.numExamples because I divided the slack contribution weight by the number of examples.  Check that this is right.

def main():
	kernelFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/allkernels_info.txt'
	trainFile = 'train/train.newsmall_1_reducedy.txt' #TODO: Rafi, move this to your data directory so it doesn't clutter things up

	params = Params()
	params.spl_params = Params()
	params.spl_params.spl_mode = 'CCCP'
	params.max_outer_iter = 1337 #TODO: make this user input
	App.loadKernelFile(kernelFile, params) 
	loadTrainFile(trainFile, params)

	w = App.PsiObject(params,False)
	LSSVM.optimize(w, params)
 
	return params

if __name__== "__main__":
	main()
