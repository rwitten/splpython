import math
import numpy
from scipy import linalg
import sys
import random

import HImputation
import SPLInnerLoop
import SSVM

def initLatentVariables(w, params):
	for i in range(len(params.examples)):
		params.examples[i].h = random.randint(0, len(params.examples[i].hlabels) - 1)
		if params.supervised:
			assert(params.examples[i].h == 0)

def checkConvergence(w, params, curBestObj, wBest):
	obj = SSVM.computeObjective(w, params)
	if obj < curBestObj:
		wBest = w

	return (obj >= curBestObj - params.maxDualityGap, min([obj, curBestObj]), wBest)

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
		if converged:
			break

	return wBest
