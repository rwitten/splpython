import cPickle
import copy
import numpy
import os
from numpy import random
import re
from scipy import sparse
import signal
import sys
import multiprocessing#.dummy as multiprocessing

import ImageCache
import ImagePsi
import BBoxComputation

def delta(y1, y2):
	if(y1==y2):
		return 0 
	else:
		return  1

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

def SynthePsize(params, trueY, h):
	result = None
	if h == 0:
		result = sparse.dok_matrix((params.totalLength, 1))
		result[trueY + 1, 0] = params.syntheticParams.strength
	else:
		result = sparse.dok_matrix((numpy.random.randn(params.totalLength, 1)))

	result[0,0] = 1 #still have bias - might be useful, and at worst will do nothing
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
			self.fileUUID = exampleNumber
		else:
			self.processFile(inputFileLine)
		
		self.psiCache = params.cache


#	def findMVC(self,w, givenY, givenH):
#		assert(givenY == self.trueY)
#		assert(givenH == self.h)
#		
#		maxScore= float(-1e100)
#		bestH = -1
#		bestY = -1
#		for labelY in self.params.ylabels:
#			if (labelY in self.whiteList):
#				continue
#			(h, score, vec) = self.highestScoringLV(w,labelY)
#			totalScore = delta(givenY, labelY) + score
#			if totalScore >= maxScore:
#				bestH = h
#				bestY = labelY
#				maxScore = totalScore
#
#		assert(bestH > -1)
#		assert(bestY > -1)
#		const = delta(givenY, bestY)
#		vec = self.psi(givenY, givenH) - self.psi(bestY, bestH)
#		return (const,copy.deepcopy(vec) ) 
	
	def findScoreAllClasses(self, w):
		results = {}
		for label in self.params.ylabels:
			(h, score, vec) = self.highestScoringLV(w, label)
			results[label] = score
		return results

	def fillHLabels(self, filename):
		self.hlabels = []
		if self.params.supervised:
			self.hlabels.append(LatentVar(0.0, float(self.width), 0.0, float(self.height)))
		else:
			hfile = open(filename)
			for line in hfile:
				bbox_coords = line.split()
				self.hlabels.append(LatentVar(bbox_coords[0], bbox_coords[2], bbox_coords[1], bbox_coords[3]))

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
	def psi(self, y,h, doPadCanonicalPsi = True):
		psi = (self.psis()[h,:]).T
		if doPadCanonicalPsi:
			return padCanonicalPsi(psi, y, self.params)
		else:
			return psi

	def psis(self):
		if self.fileUUID in self.psiCache.map.keys():
			return self.psiCache.get(self.fileUUID)
	
		feature_cache_dir = "/vision/u/rwitten/features"
		filepath = feature_cache_dir + "/%s_%d.rlw"%(self.fileUUID, len(self.hlabels))
		if os.path.exists(filepath):
			result= ImageCache.loadObject(filepath)
			self.psiCache.set(self.fileUUID,result)
			return result

		features = []
		for h in self.hlabels:
			singleResult = None
			if self.params.syntheticParams:
				singleResult = SynthePsize(self.params, self.trueY, h)
			else:
				singleResult = OneClassPsiObject(self.params)
				for kernelNum in range(self.params.numKernels):
					for index in range(len(self.xs[kernelNum])):
						bboxesContainingDescriptor = BBoxComputation.get_bboxes_containing_descriptor(self.xs[kernelNum][index], self.ys[kernelNum][index], h)
						ImagePsi.setPsiEntry(singleResult, self.params, 0, kernelNum, bboxesContainingDescriptor, self.values[kernelNum][index],1)
			features.append(singleResult)

		result = sparse.hstack( features).T.asformat('csc')
		self.psiCache.set(self.fileUUID,result)
		ImageCache.cacheObject(filepath, result)

		return result

	def highestScoringLV(self,w, labelY):
		start = labelY*self.params.totalLength
		end= (labelY+1)*self.params.totalLength
		wlocal = w[start:end,0]
		
		psis = self.psis()
		scores = psis*wlocal
		maxScore = scores.max()
		bestH = scores.argmax()

		return  (bestH, maxScore, self.psi(labelY, bestH))

def init_worker():
	return
#	signal.signal(signal.SIGINT, signal.SIG_IGN)

class FMVCJob():
	pass

def highestScoringLVUtility(w,labelY,job):
	start = labelY*job.totalLength
	end= (labelY+1)*job.totalLength
	wlocal = w[start:end,0]
	
	psis = job.psis
	scores = psis*wlocal;
	maxScore = scores.max()
	bestH = scores.argmax()

	psiVec=padCanonicalPsi((job.psis[bestH,:]).T, labelY, job)
	return (bestH, maxScore)
	
def getPsi(y,h, job):
	psi = (job.psis[h,:]).T
	return padCanonicalPsi(psi, y, job)	 

def processJob(job):
	maxScore= float(-1e100)
	bestH = -1
	bestY = -1

	for labelY in job.ylabels:
		if (labelY in job.whiteList):
			continue
		(h, score) = highestScoringLVUtility(job.w,labelY,job)
		totalScore = delta(job.givenY, labelY) + score
		if totalScore >= maxScore:
			bestH = h
			bestY = labelY
			maxScore = totalScore

	assert(bestH > -1)
	assert(bestY > -1)
	const = delta(job.givenY, bestY)
	vec = getPsi(job.givenY, job.givenH,job) - getPsi(bestY, bestH,job)
	return (const,copy.deepcopy(vec) ) 

def findCuttingPlane(w, params):
	from datetime import datetime
	start = datetime.now()

	def jobify(example):
		job = FMVCJob()
		job.psis = example.psis()
		job.whiteList =  example.whiteList
		job.ylabels = example.params.ylabels
		job.totalLength = example.params.totalLength
		job.givenY= example.trueY
		job.givenH = example.h
		job.numYLabels = example.params.numYLabels
		job.w = w
		return (job)


	def sumResults(result1, result2):
		return ( result1[0]+result2[0], result1[1]+result2[1])

	p = multiprocessing.Pool(40,init_worker)

	jobs = map(jobify, params.examples)

	try:
		output = map(processJob, jobs)
		const,vec= reduce(sumResults, output)
	except KeyboardInterrupt:
		p.terminate()
		p.join()
	else:
		p.close()
		p.join()

	end= datetime.now()
	print("FCP took %f" %( (end-start).total_seconds()))
	return (const * params.C / float(params.numExamples), vec * params.C / float(params.numExamples))

class LatentVar:
	def __init__(self, x_min, x_max, y_min, y_max):
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max
