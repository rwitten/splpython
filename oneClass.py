trainFile = 'train/train.all_1.txt'
testFile= 'train/test.all_1.txt'

def convertFile(inputFN, outputFN, label):
	fhInput = open(inputFN,'r')
	fhOutput = open(outputFN, 'w')
	topline = fhInput.readline()
	fhOutput.write(topline)


	for line in fhInput:
		splitLine = line.strip().split()
		fhOutput.write(splitLine[0]+" " + splitLine[1]+" "+ splitLine[2])
		if str(label) in splitLine[3:]:
			fhOutput.write( " 1")
		else:
			fhOutput.write( " 0")
		fhOutput.write( "\n")

for label in range(20):
	outputFNTrain = 'trainScratch/train.' + str(label)+'_1.txt'
	outputFNTest = 'trainScratch/test.' + str(label)+'_1.txt'
	convertFile(trainFile, outputFNTrain, label)
	convertFile(testFile, outputFNTest, label)

		
		
