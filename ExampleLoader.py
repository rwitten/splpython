from imageImplementation import ImageApp
from imageImplementation import SyntheticApp
import multiprocessing
from multiprocessing import Process

import math

class  ConsumerProcess( Process ):
	"""Consumes items from a Queue.

	The "target" must be a function which expects an iterable as it's
	only argument.  Therefore, the args value is not used here.
	"""
	def __init__( self, name, examples, input_queue, output_queue):
		super( ConsumerProcess, self ).__init__( name=name )
		self.input_queue= input_queue
		self.output_queue= output_queue
		self.examples= examples

	def run( self ):
		while 1:
			(blob, mapper, reducer) = self.input_queue.get()
			mapped = [mapper(blob, example) for example in self.examples]
			if reducer is not None:
				A = reduce(reducer,mapped)
				self.output_queue.put(A)
			else:
				self.output_queue.put(mapped)


#sorry for copy and pasting
def chunks(list, numChunks):
	if numChunks>len(list):
		return  [ [list[i]] for i in range(0, len(list))]
		
	chunkLength = int(math.ceil(float(len(list))/numChunks))
	return [list[i:i+chunkLength] for i in range(0, len(list), chunkLength)]

def loadExamples(params):
	if params.syntheticParams:
		examples = SyntheticApp.loadExamples(params)
	else:
		examples = ImageApp.loadExamples(params)

	numProcesses = min(multiprocessing.cpu_count(), len(examples))
	chunkedExamples = chunks(examples, numProcesses)
	params.inputQueues= [multiprocessing.Queue() for i in range(numProcesses)]
	params.outputQueue = multiprocessing.Queue()
	params.processes = []
	for i in range(numProcesses):
		p = ConsumerProcess(str(i),chunkedExamples[i] , params.inputQueues[i], params.outputQueue)
		p.start()
		params.processes.append(p)

