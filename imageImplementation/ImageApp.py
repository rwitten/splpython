import copy
import numpy
import re
from scipy import sparse
import sys

import ImagePsi
import BBoxComputation


def PsiObject(params, isFeatureVec):
	#print "params.lengthW = " + repr(params.lengthW)
	if isFeatureVec:
		po = sparse.dok_matrix( ( params.lengthW,1 ) )
		for j in range(len(params.ylabels)):
			po[j * params.totalLength, 0] = 1
	else:
		po = numpy.mat(numpy.zeros( ( params.lengthW,1 ) ))

	return po


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


class ImageExample:
	def __init__(self, inputFileLine, params, id, original_example, whiteList_swap_index):
		self.params = params
		self.id = id
		if original_example:
			print "duplicate id = " + repr(self.id) + " and refers to original id = " + repr(original_example.id)
			self.psiCache = original_example.psiCache #hopefully this will get shared properly between the two equivalent examples so we don't waste memory
			self.xs = original_example.xs #need this because different duplicates have different whitelists, so the original won't cache all its y's, plus it's bad design to assume that originals will always get their psi's cached before duplicates
			self.ys = original_example.ys
			self.values = original_example.values
			self.whiteList = copy.deepcopy(original_example.whiteList)
			self.trueY = int(original_example.whiteList[whiteList_swap_index])
			self.whiteList[whiteList_swap_index] = original_example.trueY
			self.params.hlabels = original_example.params.hlabels
			self.fileUUID = original_example.fileUUID
		else:
			print "original id = " + repr(self.id)
			self.psiCache = {}
			self.processFile(inputFileLine)

		#self.load(inputfile)

	def delta(self, y1, y2):
		if(y1==y2):
			return 0 
		else:
			return  1

	def findMVC(self,w, givenY, givenH):
		assert(givenY == self.trueY)
		assert(givenH == self.h)
		maxScore= float(-1e100)
		bestH = -1
		bestY = -1
		for labelY in self.params.ylabels:
			if (labelY in self.whiteList):
				continue
			(h, score, vec) = self.highestScoringLV(w,labelY)
			totalScore = self.delta(givenY, labelY) + score
			if totalScore >= maxScore:
				bestH = h
				bestY = labelY
				maxScore = totalScore

		assert(bestH > -1)
		assert(bestY > -1)
		const = self.delta(givenY, bestY)
		vec = self.psi(givenY, givenH) - self.psi(bestY, bestH)
		return (const,copy.deepcopy(vec) ) 
	
	def findScoreAllClasses(self, w):
		results = {}
		for label in self.params.ylabels:
			(h, score, vec) = self.highestScoringLV(w, label)
			results[label] = score
		return results

	def fill_hlabels(self, filename):
		hfile = open(filename)
		self.params.hlabels = []
		for line in hfile:
			bbox_coords = line.split()
			self.params.hlabels.append(LatentVar(bbox_coords[0], bbox_coords[2], bbox_coords[1], bbox_coords[3]))
		self.params.hlabels = self.params.hlabels[0:999:50]
		hfile.close()

	def processFile(self, inputFileLine):
		print "processFile"
		objects  = inputFileLine.split()
		self.fileUUID= objects[0]
		self.width = objects[2]
		self.height = objects[1]
		self.trueY = int(objects[3])
		self.whiteList = objects[4:]
	
		self.fill_hlabels("/afs/cs.stanford.edu/u/rwitten/scratch/mkl_features/%s.txt"%(self.fileUUID))
		
		self.xs = []
		self.ys = []
		self.values = []

		for kernelName in self.params.kernelNames:
			self.loadData(kernelName)

	def loadData(self, kernelName):
		index = len(self.xs)
		assert(index == len(self.ys))
		assert(index == len(self.values))

		self.xs.append([])
		self.ys.append([])
		self.values.append([])
		inputFile = open("/afs/cs.stanford.edu/u/rwitten/scratch/mkl_features/%s/%s_spquantized_1000_%s.mat"%(kernelName,self.fileUUID, kernelName),"r")
		
		inputFile.next() #we don't care how many indices are in the file		
		inputFile.next() #we already know the image size
		for line in inputFile:
			data = re.match('\((\d+),(\d+)\):(\d+)', line).groups()
			self.ys[index].append(int(data[0]))
			self.xs[index].append(int(data[1]))
			self.values[index].append(int(data[2])-1)

 
	# this returns a psi object
	def psi(self, y,h): 
	#	print "in psi, y = " + repr(y)
	#	print "in psi, h = " + repr(h)
		if (y,h) in self.psiCache:
			return self.psiCache[(y,h)]

		print "id = " + repr(self.id) + " y = " + repr(y) + " h = " + repr(h)
		result = PsiObject(self.params, True)
		for kernelNum in range(self.params.numKernels):
			#print "imma let ya kernel, but i just wanna say kernels kernel is the best kernel of all kernels, of all kernels!"
			for index in range(len(self.xs[kernelNum])):
				bboxes_containing_descriptor = BBoxComputation.get_bboxes_containing_descriptor(self.xs[kernelNum][index], self.ys[kernelNum][index], self.params.hlabels[h])
				ImagePsi.setPsiEntry(result, self.params, y, kernelNum, bboxes_containing_descriptor, self.values[kernelNum][index],1)
		
		self.psiCache[(y,h)] = result
		return result

	def highestScoringLV(self,w, labelY):
		maxScore = float(-1e100)
		bestH = -1
		for latentH in range(len(self.params.hlabels)):
			score = (w.T * self.psi(labelY,latentH)) [0,0]		
			if score > maxScore:
				bestH = latentH
				maxScore = score

		assert(bestH > -1)
		#print "highestScoringLV: maxScore = " + repr(maxScore)
		return (bestH, score, self.psi(labelY, bestH))

class LatentVar:
	def __init__(self, x_min, x_max, y_min, y_max):
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max

