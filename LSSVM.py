import math
import numpy
from scipy import linalg
import sys
import random
import SPLInnerLoop
import HImputation

def init_latent_variables(w, params):
	for i in range(len(params.examples)):
		params.examples[i].h = random.randrange(0, len(params.examples[i].params.hlabels) - 1)

def has_converged(w, params):
	#TODO: actually do something
	return False

def optimize(w, params):
	init_latent_variables(w, params)
	for iter in range(params.max_outer_iter):
		print "SSVM"
		w = SPLInnerLoop.optimize(w, params)
		print "imputing h"
		HImputation.impute(w, params) #this may interact with SPL at some point
		if has_converged(w, params):
			break
