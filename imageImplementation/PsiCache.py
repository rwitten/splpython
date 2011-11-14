class PsiCache(object):
	def __init__(self):
		self.map = {}

	def get(self, fileUUID):
		return self.map[fileUUID]

	def set(self, fileUUID, result):
		self.map[fileUUID] = result
