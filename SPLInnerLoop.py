import math
import numpy
from scipy import linalg
import sys
import SPLSelector
import SSVM

def optimize(w, globalSPLVars, params, iter):
	if params.splParams.splMode == 'CCCP' or iter <= params.splParams.splInitIters:
		wOptstateBundle= SSVM.cuttingPlaneOptimize(w, params, iter)
	else:
		assert(0) # I don't understand
		globalSPLVars.fraction = 1
		SPLSelector.select(globalSPLVars, w, params)
		wOptstateBundle  = SSVM.cuttingPlaneOptimize(w, params, iter)



	return wOptstateBundle
