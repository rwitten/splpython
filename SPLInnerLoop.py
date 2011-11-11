import math
import numpy
from scipy import linalg
import sys
import SSVM

def optimize(w, params):
	if params.splParams.splMode == "CCCP":
		w = SSVM.cuttingPlaneOptimize(w, params)
	elif params.splParams.splMode == 1:
		assert(0) #TODO: actually do stuff
	elif params.splParams.splMode == 2:
		assert(0) #TODO: actually do stuff
	else:
		assert(0) #TODO: actually do stuff
	return w
