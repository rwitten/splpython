from scipy import sparse

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
		return total

	def add(self, otherPsiObject, otherMultiplier):
		newObj = PsiObject(self.params)
		for labelY in self.params.ylabels:
			if labelY in self.values and labelY in otherPsiObject.values:
				newObj.set(labelY, self.values[labelY] + (otherMultiplier*otherPsiObject.values[labelY]))
			elif labelY in self.values:
				newObj.set(labelY, self.values[labelY])
			elif labelY in otherPsiObject.values:
				newObj.set(labelY, (otherMultiplier*otherPsiObject.values[labelY]))

		return newObj

	def setEntry(self, labelY, kernel, entry, value):
		if labelY not in self.values:
			self.values[labelY]= sparse.dok_matrix( (self.params.kernelLengths[kernel],1) ) 

		assert( entry<= self.params.kernelLengths[kernel])	
		self.values[labelY][entry,0] = value

	def vectorize(self):
		pass	
				
