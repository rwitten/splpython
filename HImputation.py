import copy
from scipy import sparse


from imageImplementation import CommonApp
import SSVM
import Timer


def updateRow(lilmatrix, lilrow, index):
	lilmatrix.rows[index] = lilrow.rows[0]
	lilmatrix.data[index] = lilrow.data[0]

def reoptimizeW(optState,example):
	timer = Timer.Timer(example.id)
	assert(example.params.splParams.splMode == 'CCCP')
	marginsNew = []
	margins = optState.margins
	FNewtranspose = sparse.lil_matrix(optState.F.T.shape)
	Ftranspose = optState.F.T
	Ftranspose.tolil()
	latents = optState.latents
	print("starting the magic"+ str(example.id))
	iteration=0
	timer.lap()
	ybar, hbar = latents[iteration][example.id]
	contributedMargin = CommonApp.delta(ybar, example.trueY)
	contributedConstraintGiven = CommonApp.padCanonicalPsi(example.psis()[example.h,:].T, example.trueY, example.params)
	contributedConstraintHBar = CommonApp.padCanonicalPsi(example.psis()[hbar,:].T, ybar, example.params)
	grabData = Ftranspose[iteration,:]
	computation = grabData - (contributedConstraintGiven - contributedConstraintHBar).T
	computation = computation.tolil()
	print(computation.__class__.__name__)
	timer.lap()
	updateRow(FNewtranspose, computation, iteration)
#	FNew[:,iteration] = F[:, iteration] - (contributedConstraintGiven - contributedConstraintHBar)
	timer.lap()
	marginsNew.append(margins[iteration]-contributedMargin)
	timer.lap()
	print("ending the magic"+ str(example.id))
	for iteration in range(len(margins)):
		ybar, hbar = latents[iteration][example.id]
		contributedMargin = CommonApp.delta(ybar, example.trueY)
		contributedConstraintGiven = CommonApp.padCanonicalPsi(example.psis()[example.h,:].T, example.trueY, example.params)
		contributedConstraintHBar = CommonApp.padCanonicalPsi(example.psis()[hbar,:].T, ybar, example.params)
		FNewtranspose[iteration,:] = Ftranspose[iteration,:] - (contributedConstraintGiven - contributedConstraintHBar).T
		marginsNew.append(margins[iteration]-contributedMargin)
	timer.lap()

	FNew = FNewtranspose.T
	FNew = FNew.tocsr()
	FTF = FNew.T*FNew
	env,task = SSVM.initializeMosek(example.params)
	timer.lap()

	wNew, objective, gap = SSVM.solveDualQPV2(FTF, FNew,marginsNew,example.params, env,task)
	timer.lap()

	return wNew


def imputeSingle(optState, example):
	wNew = reoptimizeW(optState,example)
	(bestH, score, psivect) = example.highestScoringLV(wNew, example.trueY)
	example.h = bestH

def impute(optState, params):
	print("Imputing H")
	CommonApp.accessExamples(params, optState, imputeSingle, None)
