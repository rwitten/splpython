from imageImplementation import CacheObj
from imageImplementation import CommonApp
import ExampleLoader
import LSSVM
import Performance
import utils
import UserInput
import SPLSelector


import sys

def main():
	try:
		params = UserInput.getUserInput('train')	
		ExampleLoader.loadExamples(params)
		CommonApp.setExampleCosts(params)
		w = None
		if params.initialModelFile:
			w = CacheObj.loadObject(params.initialModelFile)
		else:
			w = CommonApp.PsiObject(params,False)

		globalSPLVars = SPLSelector.SPLVar()
		
		if params.splParams.splMode != 'CCCP':
			SPLSelector.setupSPL(params)
	
		w = LSSVM.optimize(w, globalSPLVars, params)
		CacheObj.cacheObject(params.modelFile,w)
		Performance.printStrongAndWeakTrainError(params, w)
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

