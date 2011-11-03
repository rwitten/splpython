from scipy import sparse

def PsiObject(params):
	return sparse.dok_matrix( ( params.lengthW,1 ) )

def setPsiEntry(featureVec, params, labelY, kernel, entry, value):
	index = labelY* params.totalLength + params.kernelStarts[kernel] + entry
	assert( index <= labelY* params.totalLength + params.kernelEnds[kernel])
	featureVec[index,0] = value

				
