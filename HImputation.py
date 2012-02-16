from imageImplementation import CommonApp


def imputeSingle(w, example):
	(bestH, score, psivect) = example.highestScoringLV(w, example.trueY)
	example.h = bestH

def impute(w, params):
	print("Imputing H")
	CommonApp.accessExamples(params, w, imputeSingle, None)
