from scipy import sparse


def setPsiEntry(featureVec, params, labelY, kernel, bboxesContainingDescriptor, entry, value):
	#print "setPsiEntry"
	spmIndices= []
	if bboxesContainingDescriptor[4]:
		spmIndices.append(4)
		for i in range(4):
			if bboxesContainingDescriptor[i]:
				spmIndices.append(i)

		assert(len(spmIndices) == 2)
	else:
		spmIndices.append(5)

	for i in spmIndices:
		index = labelY * params.totalLength + params.kernelStarts[kernel] + i * params.rawKernelLengths[kernel] + entry #Note: we don't need to add an extra 1 for bias, 'cuz it's built into kernelStarts
		#print "index = " + repr(index)
		assert( index <= labelY * params.totalLength + params.kernelEnds[kernel])
		featureVec[index,0] = value
