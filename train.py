from implementation import ImageApp
from implementation import ImagePsi
import optimizer

import sys
import scipy

class Params(object):
    pass

def loadKernelFile(kernelFile, params):
	kFile = open(kernelFile, 'r')
	params.numKernels = int(kFile.readline().strip())
	params.ylabels = range(20)
	params.hlabels = [0]
	params.kernelNames = []
	params.kernelStarts = []
	params.kernelEnds = []
	params.kernelLengths= []
	params.C = 1
	params.epsilon = 0.01
	current = 0
	while 1:
		newKernelName =kFile.readline()
		if not newKernelName:
			break
		params.kernelStarts.append(current)
		params.kernelNames.append(newKernelName.strip())
		length = int(kFile.readline().strip())
		params.kernelLengths.append(length)
		params.kernelEnds.append(current+ length-1)
		current+= length

	params.totalLength = current # this is the length of w for a single class
	params.lengthW= current * len(params.ylabels)

def loadTrainFile(trainFile, params):
	tFile = open(trainFile,'r')
	params.numExamples = int(tFile.readline())
	params.examples = []
	for line in tFile:
		sys.stdout.write("%")
		params.examples.append(ImageApp.ImageExample(line,params,len(params.examples)))
	sys.stdout.write('\n')
	assert(params.numExamples == len(params.examples))
	params.maxDualityGap = params.C*params.epsilon*params.numExamples

def main():
	kernelFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/allkernels_info.txt'
	trainFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/train.newsmall_1.txt'

	params = Params()
	loadKernelFile(kernelFile, params) 
	loadTrainFile(trainFile, params)

	w = ImagePsi.PsiObject(params)
	optimizer.cuttingPlaneOptimize(w, params)
 
	return params

if __name__== "__main__":
	main()
