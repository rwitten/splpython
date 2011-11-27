from imageImplementation import CacheObj
from imageImplementation import PsiCache
import ExampleLoader
import LSSVM
import Performance
import UserInput

import os
import signal

def main():
	params = UserInput.getUserInput('test')

	ExampleLoader.loadExamples(params)

	w = CacheObj.loadObject(params.modelFile)

	CacheObj.cacheObject(params.modelFile,w)

	Performance.writePerformance(params, w, params.resultFile)

	return params

if __name__== "__main__":
	main()

