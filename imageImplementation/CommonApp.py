import cPickle
import copy
from datetime import datetime
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
import logging

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

def createFMVCJob(example,params, w=None):
	job = FMVCJob()

	#print("starting %d\n"%(example.id))
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
	job.C = params.C
	job.cost = example.cost
	job.fileUUID = example.fileUUID
	job.psis = example.psis()
	if w is not None:
		job.w = w

	return (job)

def matrixifyW(w, params):
	return w.reshape( (params.totalLength, len(params.ylabels)), order='F')

def tileW(w, params):
	wlocal = w[(params.givenY * params.totalLength):((params.givenY + 1) * params.totalLength), 0]
	return numpy.repeat(wlocal, (params.numYLabels), axis=1)

def chunks(list, numChunks):
	if numChunks>len(list):
		return  [ [list[i]] for i in range(0, len(list))]
		
	chunkLength = len(list)/numChunks
	return [list[i:i+chunkLength] for i in range(0, len(list), chunkLength)]

def sumResults(result1, result2):
	return ( result1[0]+result2[0], result1[1]+result2[1])

def accessExamples(params, blob, mapper, combiner):
	message = (blob,mapper, combiner)

	for queue in params.inputQueues:
		queue.put(message)

	output = []

	while len(output) < len(params.inputQueues):
		output.append(params.outputQueue.get())

	return output
		
	

def findCuttingPlane(w, params):
	reduce(sumResults,accessExamples(params, w, singleFMVC, sumResults))
	return
	def jobify(example):
		job = createFMVCJob(example, params)
		return (job)


	numJobs = multiprocessing.cpu_count()
	from datetime import datetime
	try:
		s1 = datetime.now()
		tasks = map(jobify, params.examples)
		tasksChunked= chunks(tasks, numJobs)
		ws = [w]*len(tasksChunked) #if len(tasks)<numJobs, numJobs!= len(tasksChunked)

		jobs = zip(tasksChunked, ws)

		s2 = datetime.now()
		print("done")
#		output = params.processPool.map(batchFMVC, jobs)
		output = map(batchFMVC, jobs)
		s3 = datetime.now()
		const,vec= reduce(sumResults, output)
		s4 = datetime.now()
	except KeyboardInterrupt:
		print "Caught KeyboardInterrupt, terminating workers"
		sys.exit(1)

#	print( "first " + str( (s2-s1).total_seconds()))
#	print( "second" + str( (s3-s2).total_seconds()))
#	print( "third " + str( (s4-s3).total_seconds()))


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

#This matrix is Hadamarded with a matrixified w; each column corresponds to a different class section of w.  In the case of SPL+, all columns of this matrix are the same.
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
		
def singleFMVC(w, example):
	#print("enter singleFMVC\n")
	if job.splMode == 'SPL' and job.localSPLVars.selected == 0.0:
		return (0.0, sparse.dok_matrix((job.totalLength * job.numYLabels, 1)), 0.0)

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

	vMask = numpy.asmatrix((splMat[:,bestY])).T
	canonicalPsiGiven = sparse.csr_matrix(numpy.multiply(vMask, (job.psis[job.givenH,:].T).todense() ))
	canonicalPsiBest = sparse.csr_matrix(numpy.multiply(vMask, (job.psis[bestH,:].T).todense() ))
	
	psiGiven = padCanonicalPsi(canonicalPsiGiven, job.givenY, job)
	psiBest = padCanonicalPsi(canonicalPsiBest, bestY, job)

	psiVec = psiGiven - psiBest
	return (job.cost * const, job.cost * copy.deepcopy(psiVec), job.cost * topScore)


def getFilepath(example):
	feature_cache_dir = "/vision/u/rwitten/features"
	if example.params.babyData == 1:
		return feature_cache_dir + "/%s_100.rlw"%(example.fileUUID)
	else:
		return feature_cache_dir + "/%s_%d.rlw"%(example.fileUUID, len(example.hlabels))
	

def tryGetFromCache(example):
	filepath = getFilepath(example)
	if os.path.exists(filepath):
#		result = CacheObj.loadObject(filepath)
		try:
			result = CacheObj.loadObject(filepath)
		except:
			sys.stdout.write("some sort of disk corruption\n")
			sys.stdout.flush()
			return (None, False)
		sys.stdout.write("%")
		sys.stdout.flush()
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
