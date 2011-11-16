from imageImplementation import CacheObj
from imageImplementation import ImageApp as App
from imageImplementation import PsiCache
import LSSVM
import UserInput
import utils

import os
import signal

def main():
	params = UserInput.getUserInput('train')
	
	if params.syntheticParams:
		utils.synthesizeExamples(params)
	else:
		App.loadKernelFile(params.kernelFile, params)
		utils.loadDataFile(params.dataFile, params)

	w = App.PsiObject(params,False)
	w= LSSVM.optimize(w, params)

	CacheObj.cacheObject(params.modelFile,w)

	utils.printStrongAndWeakTrainError(params, w)

if __name__== "__main__":
	main()

