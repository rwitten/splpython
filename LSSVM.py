import math
import numpy
from scipy import linalg
import sys
import random

import HImputation
import SPLInnerLoop
import SSVM

def writeModel(params, modelFile, w):
	mf = open(modelFile, "w")
	mf.write("%d "%(params.lengthW))
	for i in range(params.lengthW - 1):
		mf.write("%lf "%(w[i, 0]))

	mf.write("%lf"%(w[params.lengthW - 1, 0]))
	mf.close()

def initLatentVariables(w, params):
	for i in range(len(params.examples)):
		params.examples[i].h = random.randint(0, len(params.examples[i].hlabels) - 1)
		if params.supervised:
			assert(params.examples[i].h == 0)

def checkConvergence(w, params, curBestObj, wBest):
	obj,margin,constraint = SSVM.computeObjective(w, params)

	if obj < curBestObj:
		wBest = w
		

	return (obj >= (curBestObj - params.maxDualityGap), min([obj, curBestObj]), wBest)

def optimize(w, params):
	bestObj = numpy.inf
	wBest = w
	initLatentVariables(w, params)
	for iter in range(params.maxOuterIters):
		print("SSVM iteration %d"  % (iter))
		w = SPLInnerLoop.optimize(w, params)
		print("Imputing h")
		HImputation.impute(w, params) #this may interact with SPL at some point
		(converged, bestObj, wBest) = checkConvergence(w, params, bestObj, wBest)
		if converged or params.supervised:
			break
		writeModel(params, params.modelFile + str(iter), w)
		

	return wBest
