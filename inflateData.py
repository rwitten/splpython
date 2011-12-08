import sys
import os
import copy

#Usage: python inflateData.py <name of input file> <name of output file>

def swap(suff, i, j):
	suffcopy = copy.deepcopy(suff)
	temp = suffcopy[i]
	suffcopy[i] = suffcopy[j]
	suffcopy[j] = temp
	return suffcopy

def getAllOrderings(suff):
	allsuffs = []
	for j in range(len(suff)):
		allsuffs.append(swap(suff, 0, j))

	return allsuffs

def inflateData(data):
	newData = []
	for datum in data:
		suff = datum[1]
		allsuffs = getAllOrderings(suff)
		for newsuff in allsuffs:
			newData.append((datum[0], newsuff))

	return newData

def readData(filename):
	file = open(filename, "r")
	file.readline()
	origData = []
	for line in file:
		objects = line.split()
		pref = objects[0:3]
		suff = []
		objects = objects[3:]
		for i in range(len(objects)):
			suff.append(int(objects[i]))

		origData.append((pref, suff))

	file.close()
	return origData

def writeData(filename, data):
	file = open(filename, "w")
	file.write("%d\n"%(len(data)))
	for datum in data:
		str = "%s %s %s"%(datum[0][0], datum[0][1], datum[0][2])
		for label in datum[1]:
			str = "%s %d"%(str, label)

		str = "%s\n"%(str)
		file.write(str)

	file.close()

infilename = sys.argv[1]
outfilename = sys.argv[2]
origData = readData(infilename)
newData = inflateData(origData)
writeData(outfilename, newData)
