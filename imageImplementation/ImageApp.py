import cPickle
import copy
from datetime import datetime
import multiprocessing
import numpy
from numpy import random
import os
import re
from scipy import sparse
import sys

import BBoxComputation
import CommonApp
import ImageCache
import ImagePsi
import PsiCache

class ImageExample:
	def __init__(self, params, exampleNumber, input):
		self.params = params
		self.id = exampleNumber
		self.processFile(input)
		self.psiCache = params.cache


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

			self.hlabels = self.hlabels[:100]

	def processFile(self, inputFileLine):
		print("analyzing " + inputFileLine)
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
			return CommonApp.padCanonicalPsi(psi, y, self.params)
		else:
			return psi

	def psis(self):
		(result, success) = CommonApp.tryGetFromCache(self)
		if success:
			return result

		features = []
		for h in self.hlabels:
			singleResult = CommonApp.OneClassPsiObject(self.params)
			for kernelNum in range(self.params.numKernels):
				for index in range(len(self.xs[kernelNum])):
					bboxesContainingDescriptor = BBoxComputation.get_bboxes_containing_descriptor(self.xs[kernelNum][index], self.ys[kernelNum][index], h)
					ImagePsi.setPsiEntry(singleResult, self.params, 0, kernelNum, bboxesContainingDescriptor, self.values[kernelNum][index],1)
			features.append(singleResult)

		result = sparse.hstack( features).T.asformat('csc')
		CommonApp.putInCache(self, result)
		return result

	def highestScoringLV(self,w, labelY):
		return CommonApp.highestScoringLVGeneral(w,labelY,self.params,self.psis())

class LatentVar:
        def __init__(self, x_min, x_max, y_min, y_max):
                self.x_min = x_min
                self.x_max = x_max
                self.y_min = y_min
                self.y_max = y_max


def loadKernelFile(params):
        kFile = open(params.kernelFile, 'r')
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

def loadDataFile(params):
	tFile = open(params.dataFile,'r')
	params.numExamples= int(tFile.readline())
	params.examples = []
	params.cache= PsiCache.PsiCache()
	params.processQueue = multiprocessing.Pool(40)

	for line in tFile:
		sys.stdout.write("%")
		sys.stdout.flush()
		params.examples.append(ImageExample(params, len(params.examples), line))

	print "total number of examples (including duplicates) = " + repr(params.numExamples)
	sys.stdout.write('\n')
	assert(params.numExamples == len(params.examples))

def loadExamples(params):
	loadKernelFile(params)
	loadDataFile(params)
