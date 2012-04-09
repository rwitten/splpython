from imageImplementation import ImageApp
from imageImplementation import SyntheticApp
import logging
import multiprocessing
from multiprocessing import Process
import HImputation
import SSVM
#from multiprocessing import dummy as multiprocessing
#from multiprocessing.dummy import Process

import math
import os
import sys

class  ConsumerProcess( Process ):
	"""Consumes items from a Queue.
	"""
	def __init__( self, name, examples,fileScratchName,  input_queue, output_queue):
		super( ConsumerProcess, self ).__init__( name=name )
		self.input_queue= input_queue
		self.output_queue= output_queue
		self.examples= examples
		self.fileScratchName = fileScratchName
		self.name = name
		
	def run( self ):
		try:
			while 1:
				(blob, mapper, reducer) = self.input_queue.get()
				if mapper == HImputation.imputeSingle:
					env,task = SSVM.initializeMosek(self.examples[0].params)
					blob.env = env
					blob.task = task

				mapped = [mapper(blob,example) for example in self.examples]
				if reducer is not None:
					A = reduce(reducer,mapped)
					self.output_queue.put(A)
				else:
					self.output_queue.put(mapped)

		except Exception, e:
			logging.debug('worker died')
			import traceback
			traceback.print_exc(file=sys.stderr)
			sys.stderr.flush()


#sorry for copy and pasting
def chunks(inputList, numChunks):
	if numChunks==0:
		assert(len(inputList)==0)
		return []

	if numChunks>len(inputList):
		return  [ [inputList[i]] for i in range(0, len(inputList))]
	
	chunkLength = int(math.ceil(float(len(inputList))/numChunks))
	return [inputList[:chunkLength]] + chunks(inputList[chunkLength:], numChunks-1)

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
	try:
		os.mkdir('%sprocesses'%params.scratchFile)
	except OSError, e:
		pass

	for i in range(numProcesses):
		p = ConsumerProcess(str(i),chunkedExamples[i],params.scratchFile, params.inputQueues[i], params.outputQueue)
		p.start()
		params.processes.append(p)

