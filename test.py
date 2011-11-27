from imageImplementation import CacheObj
from imageImplementation import ImageApp as App
from imageImplementation import PsiCache
import LSSVM
import UserInput
import utils

import os
import signal

def main():
	params = UserInput.getUserInput('test')
	if params.syntheticParams:
		utils.synthesizeExamples(params)
	else:
		App.loadKernelFile(params.kernelFile, params)
		utils.loadDataFile(params.dataFile, params)

	w = CacheObj.loadObject(params.modelFile)

	CacheObj.cacheObject(params.modelFile,w)

	utils.writePerformance(params, w, params.resultFile)

	return params

if __name__== "__main__":
	main()

