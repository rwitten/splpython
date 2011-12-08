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
	if params.splParams.splMode != 'CCCP':
		job.localSPLVars = example.localSPLVars

	#print("ending %d\n"%(example.id))
	job.whiteList =  example.whiteList
	job.ylabels = example.params.ylabels
	job.totalLength = example.params.totalLength
	job.kernelLengths = params.kernelLengths
	job.splMode = params.splParams.splMode
	job.givenY= example.trueY
	job.givenH = example.h
	job.numYLabels = example.params.numYLabels
	job.w = w
	job.C = params.C
	job.fileUUID = example.fileUUID
	return (job)

def matrixifyW(w, params):
	return w.reshape( (params.totalLength, len(params.ylabels)), order='F')

def tileW(w, params):
	wlocal = w[(params.givenY * params.totalLength):((params.givenY + 1) * params.totalLength), 0]
	return numpy.repeat(wlocal, (params.numYLabels), axis=1)

def findCuttingPlane(w, params):
	def jobify(example):
		job = createFMVCJob(example, w, params)
		return (job)

	def sumResults(result1, result2):
		return ( result1[0]+result2[0], result1[1]+result2[1])

	from datetime import datetime
	try:
		s1 = datetime.now()
		tasks = map(jobify, params.examples)
		s2 = datetime.now()
		#output = params.processQueue.map(singleFMVC, tasks)
		output = map(singleFMVC, tasks)
		s3 = datetime.now()
		const,vec= reduce(sumResults, output)
		s4 = datetime.now()
	except KeyboardInterrupt:
		print "Caught KeyboardInterrupt, terminating workers"
		sys.exit(1)

	print( "first " + str( (s2-s1).total_seconds()))
	print( "second" + str( (s3-s2).total_seconds()))
	print( "third " + str( (s4-s3).total_seconds()))


	const = const/len(params.examples)
	vec = vec/len(params.examples)

	return (const, vec)

def highestScoringLVGeneral(w, labelY, params, psis):
	start = labelY*params.totalLength
	end= (labelY+1)*params.totalLength
	wlocal = w[start:end,0]

	#print("height of wlocal is %d\n"%(wlocal.shape[0]))
	
	scores = psis*wlocal;
	maxScore = scores.max()
	bestH = scores.argmax()
	
	psiVec=padCanonicalPsi((psis[bestH,:]).T, labelY, params)

	return (bestH, maxScore,psiVec)

def highestScoringLVUtility(w, labelY, job, splMat):
	modifiedPsis = job.psis * splMat
	return highestScoringLVGeneral(w, labelY, job, modifiedPsis)
		
def getPsi(y,h, job, splMat):
	psi = splMat * (job.psis[h,:]).T
	return padCanonicalPsi(psi, y, job)

def createDeltaVec(job):
	deltaVec = None
	if job.splMode == 'SPL' or job.splMode == 'CCCP':
		deltaVec = numpy.ones((1, job.numYLabels))
	elif job.splMode == 'SPL+':
		deltaVec = numpy.repeat(numpy.array([numpy.mean(job.localSPLVars.selected, axis=1)]), (job.numYLabels), axis=1)
	elif job.splMode == 'SPL++':
		deltaVec = numpy.zeros((1, job.numYLabels))
		for labelY in range(job.numYLabels):
			deltaVec[0, labelY] = numpy.mean(job.localSPLVars.selected[labelY], axis = 1)
	else:
		assert(0)

	for labelY in range(job.numYLabels):
		deltaVec[0, labelY] = deltaVec[0, labelY] * delta(job.givenY, labelY)

	return deltaVec

def createSPLMat(job):
	if job.splMode == 'SPL' or job.splMode == 'CCCP':
		return numpy.ones((job.totalLength, job.numYLabels))
	elif job.splMode == 'SPL+':
		selectionVec = (job.localSPLVars.selected).T
		columnVec = numpy.concatenate((numpy.ones((1,1)), numpy.repeat(selectionVec, job.kernelLengths, axis=0)), axis=0)
		return numpy.repeat(columnVec, (job.numYLabels), axis=1)
	elif job.splMode == 'SPL++':
		columnVecs = []
		for labelY in range(job.numYLabels):
			selectionVec = (job.localSPLVars.selected[labelY]).T
			columnVecs.append(numpy.concatenate((numpy.ones((1,1)), numpy.repeat(selectionVec, job.kernelLengths, axis=0)), axis=0))

		return numpy.concatenate(columnVecs, axis=1)
	else:
		assert(0)
		return None
		
def singleFMVC(job):
	#print("enter singleFMVC\n")
	if job.splMode == 'SPL' and job.localSPLVars.selected == 0.0:
		return (0.0, sparse.dok_matrix((job.totalLength, 1)), 0.0)

	splMat = createSPLMat(job)
	fullWMat = matrixifyW(job.w, job)
	givenWMat = tileW(job.w, job)
	deltaVec = createDeltaVec(job)
	A = splMat * numpy.array(fullWMat)
	B = splMat * numpy.array(givenWMat)
	violatorScores = job.psis * numpy.matrix(A)
	givenScores = job.psis[job.givenH,:] * numpy.matrix(B)
	topViolatorScores = violatorScores.max(0)
	topViolatorScorers = violatorScores.argmax(0)
	totalScores = topViolatorScores + deltaVec - givenScores
	for whiteLabel in job.whiteList:
		totalScores[0, whiteLabel] = -1e10 #This ensures that none of the labels in the whitelist will get chosen as the MVC

	topScore = (totalScores.max(1))[0,0]
	assert(topScore >= -1e-5)
	bestY = (totalScores.argmax(1))[0,0]
	assert(bestY not in job.whiteList)
	bestH = topViolatorScorers[0, bestY]
	const = deltaVec[0, bestY]
	#print("splMat is %d by %d\n"%(splMat.shape[0], splMat.shape[1]))
	diagVec = numpy.zeros((1, job.totalLength))
	diagVec[0,:] = (splMat[:,bestY]).T
	splDiag = sparse.spdiags(diagVec, numpy.array([0]), job.totalLength, job.totalLength)
	canonicalPsiBest = splDiag * (job.psis[bestH,:]).T
	#print("canonicalPsiBest is %d by %d\n"%(canonicalPsiBest.shape[0], canonicalPsiBest.shape[1]))
	canonicalPsiGiven = splDiag * (job.psis[job.givenH,:]).T
	psiBest = padCanonicalPsi(canonicalPsiBest, bestY, job)
	psiGiven = padCanonicalPsi(canonicalPsiGiven, job.givenY, job)
	psiVec = psiGiven - psiBest
	#print("exit singleFMVC\n")
	return (const, copy.deepcopy(psiVec), topScore)

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
#		result = CacheObj.loadObject(filepath)
		try:
			result = CacheObj.loadObject(filepath)
		except:
			sys.stdout.write("some sort of disk corruption\n")
			sys.stdout.flush()
			return (None, False)
		example.psiCache.set(example.fileUUID,result)
		sys.stdout.write("%")
		sys.stdout.flush()
		return (result, True)

	return (None, False)

def putInCache(example, result):
	filepath = getFilepath(example)
	example.psiCache.set(example.fileUUID, result)
	CacheObj.cacheObject(filepath, result)
