import math
import numpy
from scipy import linalg
import sys

def impute(w, params, spl_params):
	for i in range(len(params.examples)):
		old_psivect = params.examples[i].psi(params.examples[i].trueY, params.examples[i].h)
		old_score = (w.T * old_psivect)[0,0]
		(bestH, score, psivect) = params.examples[i].highestScoringLV(w, params.examples[i].trueY)
		if params.examples[i].h != bestH:
			print "example " + repr(i) + " changed its h from " + repr(params.exmaples[i].h) + " to " + repr(bestH)
			print "score changed from " + repr(old_score) + " to " + repr(score)

		params.examples[i].h = bestH
