import math
import numpy
from scipy import linalg
import sys
import random


import HImputation
from imageImplementation import CacheObj
import SPLInnerLoop
import SSVM

def initLatentVariables(w, params):
	for i in range(len(params.examples)):
		params.examples[i].h = random.randint(0, len(params.examples[i].hlabels) - 1)
		if params.supervised:
			assert(params.examples[i].h == 0)
	if params.initialModelFile:
		HImputation.impute(w, params)

def checkConvergence(w, globalSPLVars, params, curBestObj, wBest):
	obj,margin,constraint = SSVM.computeObjective(w, params)

	if obj < curBestObj:
		wBest = w
		
	return (obj >= (curBestObj - params.maxDualityGap), min([obj, curBestObj]), wBest)

def optimize(w, globalSPLVars, params):
	random.seed(params.seed)
	bestObj = numpy.inf
	initLatentVariables(w, params)
	for iter in xrange(params.maxOuterIters):
		print("SSVM iteration %d"  % (iter))
		w = SPLInnerLoop.optimize(w, globalSPLVars, params, iter)
		print("Imputing h")
		HImputation.impute(w, params) #this may interact with SPL at some point
		(converged, bestObj, w) = checkConvergence(w, globalSPLVars, params, bestObj, w)
		if converged and iter > params.splParams.splInitIters and globalSPLVars.fraction >= 1.0:
			print("Breaking because of convergence")
			break
		elif params.supervised:
			print("Only one run because we are in supervised mode")
			break
		
		CacheObj.cacheObject(params.modelFile + "."+str(iter), w)
		

	return w
