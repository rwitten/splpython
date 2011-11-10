import math
import numpy
from scipy import linalg
import sys
import SSVM

def optimize(w, params):
	if params.spl_params.spl_mode == "CCCP":
		w = SSVM.cuttingPlaneOptimize(w, params)
	elif params.spl_params.spl_mode == 1:
		assert(0) #TODO: actually do stuff
	elif params.spl_params.spl_mode == 2:
		assert(0) #TODO: actually do stuff
	else:
		assert(0) #TODO: actually do stuff
	return w
