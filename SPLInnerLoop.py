import math
import numpy
from scipy import linalg
import sys
import SPLSelector
import SSVM

def optimize(w, globalSPLVars, params, iter):
	if params.splParams.splMode == 'CCCP' or iter < params.splParams.splInitIters:
		w = SSVM.cuttingPlaneOptimize(w, params, iter)
	else:
		globalSPLVars.fraction = min(params.splParams.splInitFraction + params.splParams.splIncrement * float(iter - params.splParams.splInitIters), 1.0)
		for i in range(params.splParams.splOuterIters):
			print("SPL outer iter %d\n"%(i))
			SPLSelector.select(globalSPLVars, w, params)
			w = SSVM.cuttingPlaneOptimize(w, params, iter)

	return w
