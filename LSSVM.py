import math
import numpy
from scipy import linalg
import sys
import utils
import random

from imageImplementation import CommonApp
import HImputation
from imageImplementation import CacheObj
import SPLInnerLoop
import SSVM


def initLatentVariables(w, params):
	if params.initialModelFile:
		HImputation.impute(w, params)
	
def checkConvergence(w, globalSPLVars, params, curBestObj, wBest,iter):
	if (params.splParams.splMode!='CCCP') and globalSPLVars.fraction < .9999: #NOT INCLUDING EVERYTHING, IGNORE RESULTS
		return ( False, numpy.inf, w)


	obj,margin,constraint,mostViolatedLatents= SSVM.computeObjective(w, params)

	if obj < curBestObj:
		wBest = w

	if params.splParams.splControl and params.splParams.splMode == 'CCCP' and iter < 5: #WE USE 5 ITERATIONS TO ACHIEVE COMPUTATIONAL PARITY
		return (False, min([obj, curBestObj]), wBest)

	return (obj >= (curBestObj - params.maxDualityGap), min([obj, curBestObj]), wBest)

def optimize(w, globalSPLVars, params):
	bestObj = numpy.inf
	initLatentVariables(w, params)
	utils.dumpCurrentLatentVariables(params, "%s.%s"%(params.latentVariableFile, 'init'))
	for iter in xrange(params.maxOuterIters):
		print("SSVM iteration %d"  % (iter))
		w,optState = SPLInnerLoop.optimize(w, globalSPLVars, params, iter)
		print("Imputing h")
		HImputation.impute(optState, params) 
		(converged, bestObj, w) = checkConvergence(w, globalSPLVars, params, bestObj, w,iter)
		print("Best obj %f" % bestObj)
		if converged and ((params.splParams.splMode=='CCCP') or (iter > params.splParams.splInitIters and globalSPLVars.fraction >= 1.0)):
			print("Breaking because of convergence")
			break
		elif params.supervised:
			print("Only one run because we are in supervised mode")
			break
		
		CacheObj.cacheObject(params.modelFile + "."+str(iter), w)
		utils.dumpCurrentLatentVariables(params, "%s.%d"%(params.latentVariableFile, iter))
	utils.dumpCurrentLatentVariables(params, params.latentVariableFile)
	print("Best objective attained is %f" % bestObj)

	return w
