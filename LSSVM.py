import logging
import math
import numpy
from scipy import linalg
import sys
import utils
import random

from imageImplementation import CommonApp
from imageImplementation import CacheObj
import HImputation
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

	return (obj >= (curBestObj - params.maxDualityGap), min([obj, curBestObj]), wBest,obj)

def optimize(w, globalSPLVars, params):
	bestObj = numpy.inf
	wBest= CommonApp.PsiObject(params,False)

	initLatentVariables(w, params)
	utils.dumpCurrentLatentVariables(params, "%s.%s"%(params.latentVariableFile, 'init'))
	iter = 0
	while True:
		logging.debug("SSVM iteration %d"  % (iter))

		w,optState = SPLInnerLoop.optimize(w, globalSPLVars, params, iter)
		(converged, bestObj, wBest,newObj) = checkConvergence(w, globalSPLVars, params, bestObj, w,iter)
		logging.debug("Objective after optimizing w: %f" % newObj)

		lastedLongEnough= (iter > params.minOuterIters) and ((params.splParams.splMode=='CCCP' ) or (params.splParams.splMode!='CCCP' and iter > params.splParams.splInitIters and globalSPLVars.fraction >= 1.0))
		logging.debug("Breaking because of convergence")
		if (converged and lastedLongEnough):
			logging.debug("Breaking because of convergence")
			break
		elif iter>params.maxOuterIters:
			logging.debug("Breaking because its been 10 iterations")
			break
		elif params.supervised:
			logging.debug("Only one run because we are in supervised mode")
			break

		HImputation.impute(optState, params) 
		(converged, bestObj, wBest,newObj) = checkConvergence(w, globalSPLVars, params, bestObj, w,iter)
		logging.debug("Objective after updating latents: %f" % newObj)

		logging.debug("Objective (best so far) %f" % bestObj)

		
		CacheObj.cacheObject(params.modelFile + "."+str(iter), w)
		utils.dumpCurrentLatentVariables(params, "%s.%d"%(params.latentVariableFile, iter))
		iter += 1

	utils.dumpCurrentLatentVariables(params, params.latentVariableFile)
	logging.debug("Best objective attained is %f" % bestObj)

	return wBest

