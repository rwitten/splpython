from implementation import ImageApp
from implementation import ImagePsi
import LSSVM

import sys
import scipy

class Params(object):
    pass

def loadKernelFile(kernelFile, params):
	kFile = open(kernelFile, 'r')
	params.numKernels = int(kFile.readline().strip())
	params.ylabels = range(5)
	params.hlabels = [0]
	params.kernelNames = []
	params.kernelStarts = []
	params.kernelEnds = []
	params.kernelLengths= []
	params.rawKernelLengths = []
	params.C = 10
	params.epsilon = 0.01
	current = 1 #to account for bias
	while 1:
		newKernelName =kFile.readline()
		if not newKernelName:
			break
		params.kernelStarts.append(current)
		params.kernelNames.append(newKernelName.strip())
		rawLength = int(kFile.readline().strip())
		params.rawKernelLengths.append(rawLength)
		length = 6 * rawLength #to account for SPM (plus whole image plus outside of bbox)
		params.kernelLengths.append(length)
		params.kernelEnds.append(current + length-1)
		current += length

	params.totalLength = current # this is the length of w for a single class
	params.lengthW= current * len(params.ylabels)

def loadTrainFile(trainFile, params):
	tFile = open(trainFile,'r')
	params.numUniqueExamples = int(tFile.readline())
	params.examples = []
	for line in tFile:
		sys.stdout.write("%")
		params.examples.append(ImageApp.ImageExample(line, params, len(params.examples), None, -1))
	
	for i in range(params.numUniqueExamples):
		for j in range(len(params.examples[i].whiteList)):
			params.examples.append(ImageApp.ImageExample(None, params, len(params.examples), params.examples[i], j))

	params.numExamples = len(params.examples)
	print "total number of examples (including duplicates) = " + repr(params.numExamples)
	sys.stdout.write('\n')
	assert(params.numExamples == len(params.examples))
	params.maxDualityGap = params.C*params.epsilon #*params.numExamples #TODO: I took away params.numExamples because I divided the slack contribution weight by the number of examples.  Check that this is right.

def main():
	kernelFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/allkernels_info.txt'
	trainFile = './train.newsmall_1_reducedy.txt' #TODO: Rafi, move this to your data directory so it doesn't clutter things up

	params = Params()
	spl_params = Params()
	spl_params.spl_mode = 0
	params.max_outer_iter = 1337 #TODO: make this user input
	loadKernelFile(kernelFile, params) 
	loadTrainFile(trainFile, params)

	w = ImagePsi.PsiObject(params)
	LSSVM.optimize(w, params, spl_params)
 
	return params

if __name__== "__main__":
	main()
