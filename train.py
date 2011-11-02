from implementation import ImageApp
from implementation import ImagePsi

import sys

class Params(object):
    pass

def loadKernelFile(kernelFile, params):
	kFile = open(kernelFile, 'r')
	params.numKernels = int(kFile.readline().strip())
	params.ylabels = range(20)
	params.kernelNames = []
	params.kernelStarts = []
	params.kernelEnds = []
	params.kernelLengths= []
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

	params.totalLength = current

def loadTrainFile(trainFile, params):
	tFile = open(trainFile,'r')
	params.numExamples = int(tFile.readline())
	params.examples = []
	for line in tFile:
		params.examples.append(ImageApp.ImageExample(line,params,len(params.examples)))
	sys.stdout.write('\n')
	assert(params.numExamples == len(params.examples))

def main():
	kernelFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/allkernels_info.txt'
	trainFile = '/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl/data/train.newsmall_1.txt'

	params = Params()
	loadKernelFile(kernelFile, params) 
	loadTrainFile(trainFile, params) 
	params.examples[0].psi(0,None)
	return params

if __name__== "__main__":
	main()

