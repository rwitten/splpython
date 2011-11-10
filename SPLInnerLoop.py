import math
import numpy
from scipy import linalg
import sys
import SSVM

def optimize(w, params, spl_params):
	if spl_params.spl_mode == 0:
		w = SSVM.cuttingPlaneOptimize(w, params, spl_params)
	elif spl_params.spl_mode == 1:
		assert(0) #TODO: actually do stuff
	elif spl_params.spl_mode == 2:
		assert(0) #TODO: actually do stuff
	else:
		assert(0) #TODO: actually do stuff
	return w
