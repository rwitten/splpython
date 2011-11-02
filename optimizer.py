from scipy import linalg

def findCuttingPlane(w,params):
	(const,vec) = params.examples[0].findMVC(w,0,None)
	notStopped = False
	while notStopped:
		for i in range(params.numExamples):
			(newConst, newVec) = params.examples[i].findMVC(w,0, None)
			const += newConst
			vec.add(newVec,1)

	return (const, vec)

def cuttingPlaneOptimize(w, params):
	constraints = []
	margins = []

	notConverged = 1
	LB = - numpy.inf
	UB = numpy.inf

	while notConverged:
		(margin, constraint) = findCuttingPlane(w, params)
		newLB = numpy.max(0,margin - w.T * constraint)
		
		if( newLB > LB):
			LB = newLB

		constraints.add(constraint)
		margins.add(margin)

		(w, newUB) = solveQP(constraints, margins)
		if(newUB < UB):
			UB = newUB
			wBest = w

		if(UB < LB + params.C * params.epsilon):
			notConverged = 0
