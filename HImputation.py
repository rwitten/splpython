import copy
import logging
import numpy
from numpy import linalg
import random 
from scipy import sparse
import sys

from imageImplementation import CommonApp
import SSVM
import Timer
import utils

def updateRow(lilmatrix, lilrow, index):
	lilmatrix.rows[index] = lilrow.rows[0]
	lilmatrix.data[index] = lilrow.data[0]

def reoptimizeW(optState,example):
	assert(example.params.splParams.splMode == 'CCCP')
	marginsNew = []
	margins = optState.margins
	FNewtranspose = numpy.asmatrix(numpy.zeros(optState.F.T.shape))
	Ftranspose = optState.F.T
	latents = optState.latents
	for iteration in range(len(margins)):
		ybar, hbar = latents[iteration][example.id]
		contributedMargin = CommonApp.delta(ybar, example.trueY)
		contributedConstraintGiven = CommonApp.padCanonicalPsi(example.psis()[example.h,:].T, example.trueY, example.params)
		contributedConstraintHBar = CommonApp.padCanonicalPsi(example.psis()[hbar,:].T, ybar, example.params)
		FNewtranspose[iteration,:] = (Ftranspose[iteration,:] - (contributedConstraintGiven - contributedConstraintHBar).T)
		marginsNew.append(margins[iteration]-contributedMargin)

	FNew = FNewtranspose.T
	FTF = FNew.T*FNew

	wNew, objective, gap = SSVM.solveDualQPV2(FTF, FNew,marginsNew,example.params, optState.env,optState.task)


	return wNew

def fractionToUCCCP(iteration):
	if iteration<4:
		return 1.0 / (iteration + .0001)
	else:
		return 0 

def imputeSingle(optState, example):
	if example.params.UCCCP:
		if random.random() < fractionToUCCCP(optState.outerIter):
			logging.debug("UCCCPing an example")
			wNew = reoptimizeW(optState,example)
		else:
			wNew = optState.w
	else:
		wNew = optState.w
	
	(bestH, score, psivect) = example.highestScoringLV(wNew, example.trueY)
	(bestHOldW, score, psivect) = example.highestScoringLV(optState.w, example.trueY)

	example.h = bestH

def impute(optState, params):
	logging.debug("Imputing H")
	CommonApp.accessExamples(params, optState, imputeSingle, None)
