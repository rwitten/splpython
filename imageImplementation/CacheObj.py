import cPickle
import zlib

def cacheObject(filename, object):
	fHandle = open(filename, 'wb')
	fHandle.write( zlib.compress(cPickle.dumps(object,cPickle.HIGHEST_PROTOCOL),9))
	fHandle.close()

def loadObject(filename):
	fHandle = open(filename, 'rb')
	resultStr = fHandle.read( )
	result = cPickle.loads(zlib.decompress(resultStr))
	fHandle.close()
	return result
	
