from imageImplementation import CacheObj
from imageImplementation import PsiCache
import ExampleLoader
import LSSVM
import Performance
import UserInput

import os
import signal
import sys

def main():
	try:
		params = UserInput.getUserInput('test')
		ExampleLoader.loadExamples(params)
		w = CacheObj.loadObject(params.modelFile)
		Performance.writePerformance(params, w, params.resultFile)
		Performance.printStrongAndWeakTrainError(params, w)
		utils.dumpCurrentLatentVariables(params, params.latentVariableFile)
	except Exception, e :
		import traceback
		traceback.print_exc(file=sys.stdout)
	finally:
		if params.processes is not None:
			for p in params.processes:
				p.terminate()
		sys.exit(1)


if __name__== "__main__":
	main()

