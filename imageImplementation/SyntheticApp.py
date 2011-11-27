import cPickle
import copy
from datetime import datetime
import numpy
from numpy import random
import os
import re
from scipy import sparse
import sys

import CommonApp
import ImagePsi
import PsiCache

class SyntheticExample:
	def __init__(self, params, exampleNumber, input):
		self.params = params
		self.id = exampleNumber
		self.whiteList = []
		self.hlabels = range(params.syntheticParams.numLatents)
		self.trueY = exampleNumber % len(params.ylabels)
		self.fileUUID = exampleNumber
		self.psiCache = params.cache


	def findScoreAllClasses(self, w):
		results = {}
		for label in self.params.ylabels:
			(h, score, vec) = self.highestScoringLV(w, label)
			results[label] = score
		return results

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
			singleResult = None
			if self.params.syntheticParams:
				singleResult = SynthePsize(self.params, self.trueY, h)
			
			features.append(singleResult)

		result = sparse.hstack( features).T.asformat('csc')
		CommonApp.putInCache(self, result)
		return result

	def highestScoringLV(self,w, labelY):
		return CommonApp.highestScoringLVGeneral(w,labelY,self.params,self.psis())

def SynthePsize(params, trueY, h):
	result = None
	if h == 0:
		result = sparse.dok_matrix((params.totalLength, 1))
		result[trueY + 1, 0] = params.syntheticParams.strength
	else:
		result = sparse.dok_matrix((numpy.random.randn(params.totalLength, 1)))

	result[0,0] = 1 #still have bias - might be useful, and at worst will do nothing
	return result

def loadExamples(params):
	params.cache = PsiCache.PsiCache()
	params.examples = []
	for i in range(params.numExamples):
		params.examples.append(SyntheticExample(params, i, None))
