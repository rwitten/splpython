from imageImplementation import ImageApp
from imageImplementation import SyntheticApp

def loadExamples(params):
	if params.syntheticParams:
		SyntheticApp.loadExamples(params)
	else:
		ImageApp.loadExamples(params)
