from scipy import sparse


def setPsiEntry(featureVec, params, labelY, kernel, bboxes_containing_descriptor, entry, value):
	#print "setPsiEntry"
	spm_indices = []
	if bboxes_containing_descriptor[4]:
		spm_indices.append(4)
		for i in range(4):
			if bboxes_containing_descriptor[i]:
				spm_indices.append(i)

		assert(len(spm_indices) == 2)
	else:
		spm_indices.append(5)

	for i in range(len(spm_indices)):
		index = labelY * params.totalLength + params.kernelStarts[kernel] + spm_indices[i] * params.rawKernelLengths[kernel] + entry #Note: we don't need to add an extra 1 for bias, 'cuz it's built into kernelStarts
		#print "index = " + repr(index)
		assert( index <= labelY * params.totalLength + params.kernelEnds[kernel])
		featureVec[index,0] = value
