from imageImplementation import CacheObj
<<<<<<< HEAD
from imageImplementation import ImageApp as App
from imageImplementation import PsiCache
import LSSVM
import UserInput
import utils
=======
from imageImplementation import PsiCache
import ExampleLoader
import LSSVM
import Performance
import UserInput
>>>>>>> 5631202470802a32f719785b16ed1ce51f0c37f0

import os
import signal

def main():
	params = UserInput.getUserInput('test')
	ExampleLoader.loadExamples(params)
	w = CacheObj.loadObject(params.modelFile)
	Performance.writePerformance(params, w, params.resultFile)

	return params

if __name__== "__main__":
	main()

