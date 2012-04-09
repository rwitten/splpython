import cPickle
import copy
from datetime import datetime
import logging 
import multiprocessing
import numpy
from numpy import random
import os
import random
import re
from scipy import sparse
import sys

import BBoxComputation
import CommonApp
import ImagePsi

class ImageExample:
	def __init__(self, params, exampleNumber, input):
		self.params = params
		self.id = exampleNumber
		self.processFile(input)
		self.psisMatrix = None


	def findScoreAllClasses(self, w, countDelta = False):
		results = {}
		for label in self.params.ylabels:
			(h, score, vec) = self.highestScoringLV(w, label)
			results[label] = score
			if countDelta and self.trueY != label:
				results[label] += 1
		return results

	def fillHLabels(self, filename):
		self.hlabels = []
		if self.params.supervised:
			self.hlabels.append(LatentVar(0.0, float(self.width), 0.0, float(self.height)))
		else:
			hfile = open(filename)
			for line in hfile:
				bbox_coords = line.split()
				self.hlabels.append(LatentVar(int(bbox_coords[0]), int(bbox_coords[2]), int(bbox_coords[1]), int(bbox_coords[3])))

			hfile.close()

			self.hlabels = self.hlabels[:100]
			
			if self.params.babyData == 1:
				self.hlabels = self.hlabels[:50]
			self.h = random.choice(range(len(self.hlabels)))

	def processFile(self, inputFileLine):
		objects  = inputFileLine.split()
		self.fileUUID= objects[0]
		self.width = objects[2]
		self.height = objects[1]
		self.trueY = int(objects[3])
		self.whiteList = objects[4:]
		for i in range(len(self.whiteList)):
			self.whiteList[i] = int(self.whiteList[i])
	
		self.fillHLabels("/afs/cs.stanford.edu/u/rwitten/scratch/mkl_features/%s.txt"%(self.fileUUID))
		
		self.xs = []
		self.ys = []
		self.values = []


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
		if self.psisMatrix is not None:
			return self.psisMatrix
		else:
			(result, gotIt) = CommonApp.tryGetFromCache(self)

			if gotIt:
				self.psisMatrix = result
				return result
		features = []

		for kernelName in self.params.kernelNames:
			self.loadData(kernelName)

		for h in self.hlabels:
			singleResult = CommonApp.OneClassPsiObject(self.params)
			for kernelNum in range(self.params.numKernels):
				for index in range(len(self.xs[kernelNum])):
					bboxesContainingDescriptor = BBoxComputation.get_bboxes_containing_descriptor(self.xs[kernelNum][index], self.ys[kernelNum][index], h)
					ImagePsi.setPsiEntry(singleResult, self.params, 0, kernelNum, bboxesContainingDescriptor, self.values[kernelNum][index],1)
			features.append(singleResult)

		result = sparse.hstack( features).T.asformat('csc')
		self.psisMatrix = result
		CommonApp.putInCache(self, result )
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
	examples = []

	for line in tFile:
		examples.append(ImageExample(params, len(examples), line))

	logging.debug("total number of examples (including duplicates) = " + repr(params.numExamples))
	assert(params.numExamples == len(examples))
	return examples

def loadExamples(params):
	loadKernelFile(params)
	return loadDataFile(params)
