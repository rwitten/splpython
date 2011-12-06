import cPickle
import copy
from datetime import datetime
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
		before = sparse.dok_matrix( ( params.totalLength*classY,1) )
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
	print "params.lengthW = " + repr(params.lengthW)
	if isFeatureVec:
		po = sparse.dok_matrix( ( params.lengthW,1 ) )
	else:
		po = numpy.mat(numpy.zeros( ( params.lengthW,1 ) ))

	return po

def OneClassPsiObject(params):
	result = sparse.dok_matrix( ( params.totalLength,1 ) )
	result[0,0] = 1
	return result

def createFMVCJob(example, w, params):
	job = FMVCJob()
	#print("starting %d\n"%(example.id))
	job.psis = example.psis()
	job.localSPLVars = example.localSPLVars
	#print("ending %d\n"%(example.id))
	job.whiteList =  example.whiteList
	job.ylabels = example.params.ylabels
	job.totalLength = example.params.totalLength
	job.givenY= example.trueY
	job.givenH = example.h
	job.numYLabels = example.params.numYLabels
	job.w = w
	job.C = params.C
	job.fileUUID = example.fileUUID
	return (job)

def findCuttingPlane(w, params):
	def jobify(example):
		job = createFMVCJob(example, w, params)
		return (job)

	def sumResults(result1, result2):
		return ( result1[0]+result2[0], result1[1]+result2[1])

	try:
		tasks = map(jobify, params.examples)
		output = params.processQueue.map(singleFMVC, tasks)
		#output = map(singleFMVC, tasks)
		const,vec= reduce(sumResults, output)
	except KeyboardInterrupt:
		print "Caught KeyboardInterrupt, terminating workers"
		sys.exit(1)


	

	const = const/len(params.examples)
	vec = vec/len(params.examples)

	return (const, vec)

def highestScoringLVAllClasses(w,params,psis):
	wMat = matrixifyW(w,params,psis)
	for ylabel in job.ylabels:
		if (labelY in job.whiteList) or labelY == job.givenY:
			continue
	allScores = psis*wMat

def highestScoringLVGeneral(w, labelY, params, psis):
	start = labelY*params.totalLength
	end= (labelY+1)*params.totalLength
	wlocal = w[start:end,0]
	
	scores = psis*wlocal;
	maxScore = scores.max()
	bestH = scores.argmax()
	
	psiVec=padCanonicalPsi((psis[bestH,:]).T, labelY, params)

	return (bestH, maxScore,psiVec)

def highestScoringLVUtility(w, labelY, job, splMat):
	modifiedPsis = job.psis * splMat
	return highestScoringLVGeneral(w, labelY, job.params, modifiedPsis)
		
def getPsi(y,h, job, splMat):
	psi = splMat * (job.psis[h,:]).T
	return padCanonicalPsi(psi, y, job)

def getDeltaScale(selectionVec):
	return numpy.mean(selectionVec, axis=1)

def createSPLMat(selectionVec, params):
	diagVec = numpy.concatenate(numpy.array([[1.0]]), numpy.repeat(selectionVec, params.kernelLengths), axis=1)
	splMat = sparse.spdiags(diagVec, array([0]), params.totalLength, params.totalLength)
	return splMat

def singleFMVC(job):
	splMat = sparse.eye(params.totalLength, params.totalLength)
	deltaScale = 1.0
	if job.params.splMode == 'SPL' and job.localSPLVars.selected == 0.0:
		return (0.0, sparse.dok_matrix( (params.totalLength * params.numYLabels, 1) ), 0.0)
	else if job.params.splMode == 'SPL+':
		splMat = createSPLMat(job.localSPLVars.selected, job.params)
		deltaScale = getDeltaScale(job.localSPLVars.selected)

	maxScore = -scipy.inf 
	bestH = -1
	bestY = -1
	bestSPLMat = None
	bestDeltaScale = None

	#print("starting singleFMVC()\n")

	for labelY in job.ylabels:
		if (labelY in job.whiteList):
			continue

		if job.params.splMode == 'SPL++':
			splMat = createSPLMat(job.localSPLVars.selected[labelY], job.params)
			deltaScale = getDeltaScale(job.localSPLVars.selected[labelY])

		(h, score, psiVec) = highestScoringLVUtility(job.w, labelY, job, splMat)
		score = score - (job.w.T * getPsi(job.givenY, job.givenH, job, splMat))[0,0]
		totalScore = deltaScale * delta(job.givenY, labelY) + score
		if totalScore >= maxScore:
			bestH = h
			bestY = labelY
			bestDeltaScale = deltaScale
			bestSPLMat = splMat
			maxScore = totalScore

	#print("ending singleFMVC()\n")

	assert(bestH > -1)
	assert(bestY > -1)
	const = bestDeltaScale * delta(job.givenY, bestY)
	vec = getPsi(job.givenY, job.givenH, job, bestSPLMat) - getPsi(bestY, bestH, job, bestSPLMat)
	slack = const - (job.w.T * vec)[0,0]
	assert( (const-(job.w.T*vec)[0,0]) >= -1e-3)
	return (const, copy.deepcopy(vec), slack) 

class FMVCJob():
	pass

def getFilepath(example):
	feature_cache_dir = "/vision/u/rwitten/kevin_features"
	return feature_cache_dir + "/%s_%d.rlw"%(example.fileUUID, len(example.hlabels))
	

def tryGetFromCache(example):
	if example.fileUUID in example.psiCache.map.keys():
		return (example.psiCache.get(example.fileUUID), True)

	filepath = getFilepath(example)
	if os.path.exists(filepath):
		result = CacheObj.loadObject(filepath)
		example.psiCache.set(example.fileUUID,result)
		return (result, True)

	return (None, False)

def putInCache(example, result):
	filepath = getFilepath(example)
	example.psiCache.set(example.fileUUID, result)
	CacheObj.cacheObject(filepath, result)
