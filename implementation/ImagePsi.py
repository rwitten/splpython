class PsiObject:
	def __init__(self, params):
		self.params= params
		self.values = {}

	def set(self, labelY, value):
		self.values[labelY] = value

	def dot(self, otherPsiObject):
		total = 0
		for labelY in self.values:
			if labelY in otherPsiObject.values:
				total += (self.values[labelY].T*(otherPsiObject.values[labelY]))[0,0]

	def add(self, otherPsiObject):
		newObj = PsiObject(self.params.classes)
		for labelY in self.params.classes:
			if labelY in self.values and labelY in otherPsiObject.values:
				newObj.set(labelY, self.values[labelY] + otherPsiObject.values)
			elif labelY in self.values:
				newObj.set(labelY, self.values[labelY])
			elif labelY in otherPsiObject.values:
				newObj.set(labelY, otherPsiObject.values[labelY])

		return newObj

	def setEntry(self, labelY, kernel, entry, value):
		if not self.values[labelY]:
			self.values[labelY]= scipy.sparse.dokmatrix( [self.singlePsiSize,1]) 
		
		self.values[labelY][params.kernelStart[kernel]+entry] = value
		assert( params.kernelStart[kernel]+entry <= params.kernelEnd[kernel])	

	def vectorize(self):
		pass	
				
