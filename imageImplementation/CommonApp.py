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

def findCuttingPlane(w, params):
	def jobify(example):
		job = FMVCJob()
		#print("starting %d\n"%(example.id))
		job.psis = example.psis()
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

def highestScoringLVUtility(w,labelY,job):
	return highestScoringLVGeneral(w, labelY, job, job.psis)
		
def getPsi(y,h, job):
	psi = (job.psis[h,:]).T
	return padCanonicalPsi(psi, y, job)	 

def singleFMVC(job):
	maxScore= (job.w.T*getPsi(job.givenY, job.givenH,job))[0,0]
	bestH = job.givenH
	bestY = job.givenY

	#print("starting singleFMVC()\n")

	for labelY in job.ylabels:
		if (labelY in job.whiteList) or labelY == job.givenY:
			continue
		(h, score,psiVec) = highestScoringLVUtility(job.w,labelY,job)
		totalScore = delta(job.givenY, labelY) + score
		if totalScore >= maxScore:
			bestH = h
			bestY = labelY
			maxScore = totalScore

	#print("ending singleFMVC()\n")


	assert(bestH > -1)
	assert(bestY > -1)
	const = delta(job.givenY, bestY)
	vec = getPsi(job.givenY, job.givenH,job) - getPsi(bestY, bestH,job)

	assert( (const-(job.w.T*vec)[0,0]) >= -1e-3)


	return (const,copy.deepcopy(vec) ) 

class FMVCJob():
	pass

def getFilepath(example):
	feature_cache_dir = "/vision/u/rwitten/features"
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
