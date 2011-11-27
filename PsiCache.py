import cPickle
import zlib

class PsiCache:
	def __init__(self):
		self.map = {}

	def get(self, key):
		return cPickle.loads(zlib.decompress(self.map[key]))

	def set(self, key, val):
		self.map[key] = zlib.compress(cPickle.dumps(val,cPickle.HIGHEST_PROTOCOL),9)
