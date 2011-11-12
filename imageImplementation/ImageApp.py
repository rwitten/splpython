import copy
import numpy
import os
from numpy import random
import re
from scipy import sparse
import sys

#import ImageCache #TODO: uncomment once this module exists
import ImagePsi
import BBoxComputation

def padCanonicalPsi(canonicalPsi, classY,  params):
	if ( classY > 0 and classY< params.numYLabels-1):
		before = sparse.dok_matrix( ( params.totalLength*classY,1) )
		after =sparse.dok_matrix( (params.totalLength*(params.numYLabels-classY-1), 1) )   
		return sparse.vstack( (before, canonicalPsi,  after))
	elif ( classY==0):
		after =sparse.dok_matrix( (params.totalLength*(params.numYLabels-classY-1), 1) )
		return sparse.vstack( (canonicalPsi,  after) )
	else:
		assert(classY == params.numYLabels-1)
		before = sparse.dok_matrix( ( params.totalLength*classY,1) )
		return sparse.vstack( (before,canonicalPsi))

def PsiObject(params, isFeatureVec):
	print "params.lengthW = " + repr(params.lengthW)
	if isFeatureVec:
		po = sparse.dok_matrix( ( params.lengthW,1 ) )
#		for j in range(len(params.ylabels)):
#			po[j * params.totalLength, 0] = 1
	else:
		po = numpy.mat(numpy.zeros( ( params.lengthW,1 ) ))

	return po

def OneClassPsiObject(params):
	result = sparse.dok_matrix( ( params.totalLength,1 ) )
	result[0,0] = 1
	return result

def loadKernelFile(kernelFile, params):
	kFile = open(kernelFile, 'r')
	params.numKernels = int(kFile.readline().strip())
	params.kernelNames = []
	params.kernelStarts = []
	params.kernelEnds = []
	params.kernelLengths= []
	params.rawKernelLengths = []
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
	def __init__(self, inputFileLine, params, exampleNumber):
		self.params = params
		self.id = exampleNumber
		if params.syntheticParams:
			self.whiteList = []
			self.hlabels = range(params.syntheticParams.numLatents)
			self.trueY = exampleNumber % len(params.ylabels)
			self.fileUUID = id
			return

		self.processFile(inputFileLine)
		self.psiCache = params.cache

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

	def fillHLabels(self, filename):
		hfile = open(filename)
		self.hlabels = []
		for line in hfile:
			bbox_coords = line.split()
			self.hlabels.append(LatentVar(bbox_coords[0], bbox_coords[2], bbox_coords[1], bbox_coords[3]))
		self.hlabels = self.hlabels[0:100]
		hfile.close()

	def processFile(self, inputFileLine):
		objects  = inputFileLine.split()
		self.fileUUID= objects[0]
		self.width = objects[2]
		self.height = objects[1]
		self.trueY = int(objects[3])
		self.whiteList = objects[4:]
	
		self.fillHLabels("/afs/cs.stanford.edu/u/rwitten/scratch/mkl_features/%s.txt"%(self.fileUUID))
		
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
	def psi(self, y,h, returnCanonicalPsi= False):
		if self.params.syntheticParams:
			result = None
			if y == self.trueY and h == 0:
				result = numpy.zeros((self.params.lengthW, 1))
				result[0] = self.params.syntheticParams.strength
			else:
				result = numpy.random.randn(self.params.lengthW, 1)

			return result

		if (self.fileUUID,h) in self.psiCache.map.keys():
			if returnCanonicalPsi:
				return self.psiCache.get((self.fileUUID,h))
			else:
				return padCanonicalPsi(self.psiCache.get((self.fileUUID,h)),y,self.params)

		try:
			os.makedirs("/vision/u/rwitten/features/%s" % (self.fileUUID))
		except OSError:
			pass

		filepath = "/vision/u/rwitten/features/%s/%s.rlw" %(self.fileUUID, h)
		if os.path.exists(filepath):
			result= ImageCache.loadObject(filepath)
			self.psiCache.set((self.fileUUID,h),result)
			if returnCanonicalPsi:
				return result
			else:
				return padCanonicalPsi(result, y, self.params)


		result = OneClassPsiObject(self.params)
		for kernelNum in range(self.params.numKernels):
			for index in range(len(self.xs[kernelNum])):
				bboxesContainingDescriptor = BBoxComputation.get_bboxes_containing_descriptor(self.xs[kernelNum][index], self.ys[kernelNum][index], self.hlabels[h])
				ImagePsi.setPsiEntry(result, self.params, y, kernelNum, bboxesContainingDescriptor, self.values[kernelNum][index],1)
	
		result = sparse.csr_matrix(result)	
		self.psiCache.set((self.fileUUID,h),result)
		ImageCache.cacheObject(filepath, result)

		if returnCanonicalPsi:
			return result
		else:
			return padCanonicalPsi(result, y,self.params)

	def highestScoringLV(self,w, labelY):
		maxScore = float(-1e100)
		bestH = -1
		wlocal = None
		if not self.params.syntheticParams:
			start = labelY*self.params.totalLength
			end= (labelY+1)*self.params.totalLength
			wlocal = w.T[0,start:end]
		else:
			wlocal = w.T

		for latentH in range(len(self.hlabels)):
			psiVec = self.psi(labelY,latentH, returnCanonicalPsi=True)
			score = (wlocal * psiVec) [0,0]		
			if score > maxScore:
				bestH = latentH
				maxScore = score

		assert(bestH > -1)
		return (bestH, score, self.psi(labelY, bestH))

class LatentVar:
	def __init__(self, x_min, x_max, y_min, y_max):
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max

