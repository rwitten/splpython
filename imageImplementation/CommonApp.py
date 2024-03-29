import cPickle
import copy
from datetime import datetime
import logging
import multiprocessing
import numpy
from numpy import random
import os
import re
from scipy import sparse
import signal
import sys

import CacheObj
import ImagePsi

#This is where we put stuff that's common to all applications.

def delta(y1, y2):
	if(y1==y2):
		return 0.0
	else:
		return  1.0

def padCanonicalPsi(canonicalPsi, classY,  params):
	canonicalPsi = canonicalPsi.tocoo()
	if ( classY > 0 and classY< params.numYLabels-1):
		before = sparse.dok_matrix(( params.totalLength*classY,1) )
		after =sparse.dok_matrix( (params.totalLength*(params.numYLabels-classY-1), 1) )
		start = datetime.now()
		result = sparse.vstack( (before, canonicalPsi,  after))
		end= datetime.now()

		return result
		
	elif ( classY==0):
		after =sparse.dok_matrix( (params.totalLength*(params.numYLabels-classY-1), 1) )
		return sparse.vstack( (canonicalPsi,  after) )
	else:
		assert(classY == params.numYLabels-1)
		before = sparse.dok_matrix( ( params.totalLength*classY,1) )
		return sparse.vstack( (before,canonicalPsi))

def PsiObject(params, isFeatureVec):
	logging.debug("params.lengthW = " + repr(params.lengthW))
	if isFeatureVec:
		po = sparse.dok_matrix( ( params.lengthW,1 ) )
	else:
		po = numpy.mat(numpy.zeros( ( params.lengthW,1 ) ))

	return po

def OneClassPsiObject(params):
	result = sparse.dok_matrix( ( params.totalLength,1 ) )
	result[0,0] = 1
	return result

def createFMVCJob(example,params, w=None):
	job = FMVCJob()

	if params.splParams.splMode != 'CCCP':
		job.localSPLVars = example.localSPLVars

	job.whiteList =  example.whiteList
	job.ylabels = example.params.ylabels
	job.totalLength = example.params.totalLength
	job.kernelLengths = params.kernelLengths
	job.splMode = params.splParams.splMode
	job.givenY= example.trueY
	job.givenH = example.h
	job.numYLabels = example.params.numYLabels
	job.C = params.C
	job.cost = example.cost
	job.fileUUID = example.fileUUID
	job.psis = example.psis()
	if w is not None:
		job.w = w

	return (job)

def matrixifyW(w, params):
	return w.reshape( (params.totalLength, len(params.ylabels)), order='F')

def tileW(w, example):
	params = example.params
	wlocal = w[(example.trueY * params.totalLength):((example.trueY + 1) * params.totalLength), 0]
	return numpy.repeat(wlocal, (params.numYLabels), axis=1)

def chunks(list, numChunks):
	if numChunks>len(list):
		return  [ [list[i]] for i in range(0, len(list))]
		
	chunkLength = len(list)/numChunks
	return [list[i:i+chunkLength] for i in range(0, len(list), chunkLength)]

def sumResults(result1, result2):
	return ( result1[0]+result2[0], result1[1]+result2[1],None,dict(result1[3].items()+result2[3].items()) )
	
def accessExamples(params, blob, mapper, combiner):
	logging.debug("Hitting examples with %s" % str(mapper))
	message = (blob,mapper, combiner)

	for queue in params.inputQueues:
		queue.put(message)

	output = []
	while len(output) < len(params.inputQueues):
		A = params.outputQueue.get()
		output.append(A)


	outputNew = []
	if combiner is None:
		for chunk in output:
			outputNew = outputNew +  chunk
		output = outputNew

	return output

def findCuttingPlane(w, params):
	output = accessExamples(params, w, singleFMVC, sumResults) #this gives "number of slaves" chunks
	const,vec,ignore,lvs= reduce(sumResults, output) #this makes a single guy
	const = const/params.numExamples
	vec = vec/params.numExamples
	return (const,vec,lvs)

def highestScoringLVGeneral(w, labelY, params, psis):
	start = labelY*params.totalLength
	end= (labelY+1)*params.totalLength
	wlocal = w[start:end,0]

	scores = psis*wlocal;
	maxScore = scores.max()
	bestH = scores.argmax()
	
	psiVec=padCanonicalPsi((psis[bestH,:]).T, labelY, params)

	return (bestH, maxScore,psiVec)

#def highestScoringLVUtility(w, labelY, job, splMat):
#	modifiedPsis = job.psis * splMat
#	return highestScoringLVGeneral(w, labelY, job, modifiedPsis)
#		
#def getPsi(y,h, job, splMat):
#	psi = splMat * (job.psis[h,:]).T
#	return padCanonicalPsi(psi, y, job)

def createDeltaVec(w, example):
	params = example.params
	deltaVec = None
	if params.splParams.splMode == 'SPL' or params.splParams.splMode == 'CCCP':
		deltaVec = numpy.ones((1, params.numYLabels))
	elif params.splParams.splMode == 'SPL+':
		deltaVec = numpy.repeat(numpy.array([numpy.mean(params.localSPLVars.selected, axis=1)]), (params.numYLabels), axis=1)
	elif params.splParams.splMode == 'SPL++':
		deltaVec = numpy.zeros((1, params.numYLabels))
		for labelY in range(params.numYLabels):
			deltaVec[0, labelY] = numpy.mean(params.localSPLVars.selected[labelY], axis = 1)
	else:
		assert(0)

	for labelY in range(params.numYLabels):
		deltaVec[0, labelY] = deltaVec[0, labelY] * delta(example.trueY, labelY)

	return deltaVec

#This matrix is Hadamarded with a matrixified w; each column corresponds to a different class section of w.  In the case of SPL+, all columns of this matrix are the same.
def createSPLMat(example):
	params = example.params
	if params.splParams.splMode == 'SPL' or params.splParams.splMode == 'CCCP':
		return numpy.ones((params.totalLength, params.numYLabels))
	elif params.splParams.splMode == 'SPL+':
		selectionVec = (example.localSPLVars.selected).T
		columnVec = numpy.concatenate((numpy.ones((1,1)), numpy.repeat(selectionVec, params.kernelLengths, axis=0)), axis=0)
		return numpy.repeat(columnVec, (params.numYLabels), axis=1)
	elif params.splParams.splMode== 'SPL++':
		columnVecs = []
		for labelY in range(params.numYLabels):
			selectionVec = (examples.localSPLVars.selected[labelY]).T
			columnVecs.append(numpy.concatenate((numpy.ones((1,1)), numpy.repeat(selectionVec, params.kernelLengths, axis=0)), axis=0))

		return numpy.concatenate(columnVecs, axis=1)
	else:
		assert(0)
		return None
		
def singleFMVC(w, example):
	if example.params.splParams.splMode == 'SPL' and example.localSPLVars.selected == 0.0:
		return (0.0, sparse.dok_matrix((example.params.totalLength * example.params.numYLabels, 1)), 0.0)

	splMat = createSPLMat(example)
	fullWMat = matrixifyW(w, example.params)
	givenWMat = tileW(w, example)
	deltaVec = createDeltaVec(w, example)


	assert(splMat.shape == (30001,2))
	assert(fullWMat.shape == (30001,2))
	assert(givenWMat.shape == (30001,2))
	A = splMat * numpy.array(fullWMat)
	B = splMat * numpy.array(givenWMat)

	violatorScores = example.psis() * numpy.matrix(A)
	givenScores = example.psis()[example.h,:] * numpy.matrix(B)
	topViolatorScores = violatorScores.max(0)
	topViolatorScorers = violatorScores.argmax(0)
	totalScores = topViolatorScores + deltaVec - givenScores
	for whiteLabel in example.whiteList:
		totalScores[0, whiteLabel] = -1e10 #This ensures that none of the labels in the whitelist will get chosen as the MVC

	topScore = (totalScores.max(1))[0,0]
	assert(topScore >= -1e-5)
	bestY = (totalScores.argmax(1))[0,0]
	assert(bestY not in example.whiteList)
	bestH = topViolatorScorers[0, bestY]
	const = deltaVec[0, bestY]

	vMask = numpy.asmatrix((splMat[:,bestY])).T
	canonicalPsiGiven = sparse.csr_matrix(numpy.multiply(vMask, (example.psis()[example.h,:].T).todense() ))
	canonicalPsiBest = sparse.csr_matrix(numpy.multiply(vMask, (example.psis()[bestH,:].T).todense() ))
	
	psiGiven = padCanonicalPsi(canonicalPsiGiven, example.trueY, example.params)
	psiBest = padCanonicalPsi(canonicalPsiBest, bestY, example.params)

	psiVec = psiGiven - psiBest

	lv = dict( [ (example.id, (bestY, bestH))])
	return example.cost * const, example.cost * copy.deepcopy(psiVec), example.cost * topScore,lv


def getFilepath(example):
	feature_cache_dir = "/vision/u/rwitten/features"
	if example.params.babyData == 1:
		return feature_cache_dir + "/%s_100.rlw"%(example.fileUUID)
	else:
		return feature_cache_dir + "/%s_%d.rlw"%(example.fileUUID, len(example.hlabels))
	

def tryGetFromCache(example):
	filepath = getFilepath(example)

	if os.path.exists(filepath):
		try:
			result = CacheObj.loadObject(filepath)
			logging.debug('tryGetFromCache cache %s' % example.fileUUID)
		except:
			logging.debug('tryGetFromCache corrupt weirdness %s' % example.fileUUID)
			return (None, False)
		return (result, True)

	return (None, False)

def putInCache(example, result):
	filepath = getFilepath(example)
	CacheObj.cacheObject(filepath, result)

def oneCost(ignore,x):
	x.cost = 1

def setExampleCosts(params):
	assert(params.balanceClasses!=1)
	accessExamples(params, None, oneCost, None) 
	return
	if params.balanceClasses == 1:
		counts = numpy.zeros((params.numYLabels, 1))
		for example in examples:
			counts[example.trueY, 0] += 1.0

		for example in examples:
			example.cost = float(params.numExamples) / (float(params.numYLabels) * counts[example.trueY, 0])
	else:
		for example in examples:
			example.cost = 1.0
