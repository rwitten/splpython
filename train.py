from imageImplementation import CacheObj
from imageImplementation import CommonApp
import ExampleLoader
import LSSVM
import Performance
import UserInput

def main():
	params = UserInput.getUserInput('train')	
	ExampleLoader.loadExamples(params)
	w = CommonApp.PsiObject(params,False)
	w= LSSVM.optimize(w, params)
	CacheObj.cacheObject(params.modelFile,w)
	Performance.printStrongAndWeakTrainError(params, w)

if __name__== "__main__":
	main()

