from imageImplementation import CacheObj
from imageImplementation import CommonApp
import ExampleLoader
import LSSVM
import Performance
import UserInput
import SPLSelector

def main():
	params = UserInput.getUserInput('train')	
	ExampleLoader.loadExamples(params)
	CommonApp.setExampleCosts(params)
	w = CommonApp.PsiObject(params,False)
	globalSPLVars = SPLSelector.SPLVar()
	globalSPLVars.fraction = 1.0
	if params.splParams.splMode != 'CCCP':
		SPLSelector.setupSPL(params.examples, params)

	w = LSSVM.optimize(w, globalSPLVars, params)
	CacheObj.cacheObject(params.modelFile,w)
	Performance.printStrongAndWeakTrainError(params, w)

if __name__== "__main__":
	main()

