import cPickle
import zlib

def compressObject(object):
	return zlib.compress(cPickle.dumps(object,cPickle.HIGHEST_PROTOCOL),9)

def decompressObject(resultString):
	return cPickle.loads(zlib.decompress(resultString))


def cacheObject(filename, object):
	fHandle = open(filename, 'wb')
	fHandle.write( compressObject(object))
	fHandle.close()

def loadObject(filename):
	fHandle = open(filename, 'rb')
	resultStr = fHandle.read( )
	result = decompressObject(resultStr)
	fHandle.close()
	return result
	
