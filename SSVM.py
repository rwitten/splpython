import copy 
import datetime
import math
import mosek
import numpy
from scipy import optimize
from scipy import sparse
import sys

from imageImplementation import CommonApp

#optimization problem is
# minimize .5 |w|_2^2 + C\Psi
# Psi1+F^T * w \geq delta  //matrix inequality

# dual is
# minimize .5 x^T F^T F*x + C \Psi
# Psi1 +F^T F x \geq delta 

#Psi \geq 0  

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
	pass
#	print(text)

# solves "Cutting-Plane Training of Structural SVMs", Optimization Problem 6
def solveDualQPV2(FTF, constraintsMatrix, margins, idle, params, env,task):
	def evalObjective(arg):
		arg = numpy.asmatrix(arg).T
		cost = (.5 * arg.T* FTF * arg)[0,0]
		cost += max(0,params.C* numpy.max(numpy.asmatrix(margins).T - FTF* arg ))
		return cost
	env = mosek.Env ()
	task = env.Task(0,0)
	task.putobjsense(mosek.objsense.maximize)
	NUMVAR = FTF.shape[1]
	NUMCON = 1

	task.append(mosek.accmode.var,NUMVAR)
	task.append(mosek.accmode.con,NUMCON) #1 more constraint

	for j in range(NUMVAR):
		task.putbound(mosek.accmode.var,j,mosek.boundkey.lo,0,numpy.inf)
		task.putcj(j, margins[j])
	qsubi= []
	qsubj= []
	qval= []
	for i in range(NUMVAR): #NO PSI HERE
		for j in range(i+1):
			qsubi.append(i)
			qsubj.append(j)
			qval.append(-FTF[i,j])
	task.putqobj(qsubi,qsubj,qval)

	task.putbound(mosek.accmode.con,0, mosek.boundkey.up, -numpy.inf, params.C)
	task.putavec(mosek.accmode.con,0,range(NUMVAR), [1]*NUMVAR)
	r=task.optimize()
	
	xx = numpy.zeros(NUMVAR, float)
	task.getsolutionslice(mosek.soltype.itr,
                        mosek.solitem.xx,
                        0,NUMVAR, # don't give back psi
                        xx)
	wOut = constraintsMatrix *numpy.asmatrix(xx).T
	primalObj = task.getprimalobj(mosek.soltype.itr)
	dualObj = task.getdualobj(mosek.soltype.itr)
	print("new way gets primal %f dual %f" % (primalObj, dualObj))

	return wOut,primalObj, primalObj-dualObj

def solveDualQP(FTF, constraintsMatrix, margins, idle, params, env,task):
	def evalObjective(arg):
		arg = numpy.asmatrix(arg).T
		cost = (.5 * arg.T* FTF * arg)[0,0]
		cost += max(0,params.C* numpy.max(numpy.asmatrix(margins).T - FTF* arg ))
		return cost

	#x = optimize.fmin(evalObjective, numpy.zeros( (FTF.shape[0],1) ) ) 
	#xMat = numpy.asmatrix(x).T
	#return numpy.asmatrix(constraintsMatrix*xMat), evalObjective(x),0


	env = mosek.Env ()
	task = env.Task(0,0)
	task.putobjsense(mosek.objsense.minimize)

	NUMVAR = FTF.shape[1] + 1
	NUMCON = FTF.shape[1]

	task.append(mosek.accmode.var,NUMVAR)
	task.append(mosek.accmode.con,NUMCON) #1 more constraint

	for j in range(NUMVAR-1):
		task.putbound(mosek.accmode.var,j,mosek.boundkey.ra,-1e4,1e4)
		task.putcj(j, 0)

	task.putbound(mosek.accmode.var,NUMVAR-1,mosek.boundkey.lo,0,1e4)
	task.putcj(NUMVAR-1,params.C)

	for i in range(NUMCON):
		task.putbound(mosek.accmode.con,i, mosek.boundkey.lo, margins[i], numpy.inf)
		indices = range(NUMVAR)
		constraint = (numpy.asarray(FTF[i,:])[0]).tolist() + [1]
		task.putavec(mosek.accmode.con,i,indices, constraint)

	qsubi= []
	qsubj= []
	qval= []
	for i in range(NUMVAR-1): #NO PSI HERE
		for j in range(i+1):
			qsubi.append(i)
			qsubj.append(j)
			qval.append(FTF[i,j])

	qsubi.append(NUMVAR-1)
	qsubj.append(NUMVAR-1)
	qval.append(10**-10)
	task.putqobj(qsubi,qsubj,qval)
			
	r=task.optimize()
#	print("solution status is " + str(r))

	xx = numpy.zeros(NUMVAR-1, float)
	task.getsolutionslice(mosek.soltype.itr,
                        mosek.solitem.xx,
                        0,NUMVAR-1, # don't give back psi
                        xx)

	psi= numpy.zeros(1, float)
	task.getsolutionslice(mosek.soltype.itr,
                        mosek.solitem.xx,
                        NUMVAR-1,NUMVAR, # give back psi
                        psi)

	wOut = constraintsMatrix *numpy.asmatrix(xx).T

	primalObj = task.getprimalobj(mosek.soltype.itr)
	dualObj = task.getdualobj(mosek.soltype.itr)

	print("old way gets primal %f dual %f" % (primalObj, dualObj))

	return wOut,task.getprimalobj(mosek.soltype.itr), primalObj-dualObj

def cleanUp(xx, idle, params, FTF, constraintsMatrix):
	assert((len(xx) == len(idle)) and (FTF.shape[0]==FTF.shape[1]) and (FTF.shape[0]==constraintsMatrix.shape[1]) and (len(xx)==FTF.shape[0]))
	for index in range(len(xx)):
		if xx[index]<10**-8:
			idle[index] +=1
		else:
			idle[index] = 0

	index = 0
	toDelete = []
	while index<len(idle) and len(idle)>1:
		if idle[index] > params.maxIdleIters:
			toDelete.append(index)
			del idle[index]
			FTF=numpy.delete(FTF,[index],0)
			FTF=numpy.delete(FTF,[index],1)
			constraintsMatrix =dropColumn(constraintsMatrix,index)
		else:
			index+=1
	assert((FTF.shape[0]==FTF.shape[1]) and (FTF.shape[0]==constraintsMatrix.shape[1]))


	print("CLEANUP notice: dropped %d constraints with %d left" % (len(toDelete), FTF.shape[0])) 
	return idle, FTF, constraintsMatrix

def dropColumn(constraintsMatrix, index):
	constraintsMatrix = constraintsMatrix.tocsc()
	if index == 0:
		return constraintsMatrix[:, 1:]
	elif index == constraintsMatrix.shape[1]-1:
		return constraintsMatrix[:,:-1]
	else:
		return sparse.hstack( [constraintsMatrix[:,:index-1], constraintsMatrix[:,index+1:]])	

#we want that [w \psi]^t [f_i 1] >= \delta_i

def evaluateObjectiveOnPartialQP(w, constraints, margins,params):
	psi = 0
	for i in range(len(margins)):
		psiCon = margins[i]
		for j in range(len(constraints[i][0])-1):
			psiCon = psiCon - w[constraints[i][0][j],0]*constraints[i][1][j]

		if psiCon>psi:
			psi = psiCon 

	return 0.5 * (w.T * w)[0,0] + params.C*psi


def computeObjective(w, params):
	objective = 0.5 * (w.T * w)[0,0]
	#print("w is" + repr(w.shape) )
	
	(margin, constraint) = CommonApp.findCuttingPlane(w, params)
	objective += params.C*(margin - ((w.T * constraint)[0,0]))
	return (objective, margin, constraint)


def cuttingPlaneOptimize(w, params, outerIter):
#	env,task = initializeMosek(params)
	env = mosek.Env ()
	task = env.Task(0,0)

	objective,margin, constraint = computeObjective(w, params)
	print("At beginning of iteration %f, objective = %f" % ( outerIter,objective) ) 
	F = constraint 
	FTF = (F.T*F).todense()
	
	idle = [0]
	margins = [margin]

	notConverged = 1
	LB = - numpy.inf
	UB = numpy.inf

	iter = 1
	while (UB - LB > params.maxDualityGap):
		print("Starting QP solve + constraint add")
		sys.stdout.flush()

		starttime = datetime.datetime.now()
		startqp = datetime.datetime.now()

		(w, newLB, dualityGap) = solveDualQPV2(FTF, F, margins,idle, params,env,task)
		print("Done with QP solve")
		sys.stdout.flush()

		endqp = datetime.datetime.now()	

		if (newLB > LB) and abs(dualityGap)<=params.maxDualityGap:
			LB = newLB
		
		startFMVC = datetime.datetime.now()	
		(newUB, margin, constraint) = computeObjective(w, params)
		endFMVC= datetime.datetime.now()	

		assert(margin - ((w.T * constraint)[0,0]) >= float(-1e-10)) 

		if (newUB < UB):
			UB = newUB

		idle.append(0)
		margins.append(margin)
		newPortion = (F.T*constraint).todense()
		FTF = numpy.hstack( [FTF, newPortion])
		newPortionPadded = numpy.hstack( [ newPortion.T, (constraint.T*constraint).todense()])
		FTF = numpy.vstack( [FTF, newPortionPadded])
	
		F=sparse.hstack( [F, constraint])
		endtime= datetime.datetime.now()

		print( "UB is %f and LB is %f on iteration %f" % ( UB, LB,iter) )
#		print( "New stab at UB was %f" % newUB)
		print( "TIMING step %f sec, QP took %f sec and FMVC took %f sec" % ( (endtime-starttime).total_seconds(), (endqp-startqp).total_seconds(), (endFMVC-startFMVC).total_seconds()))
		iter+=1
		sys.stdout.flush()

	objective,margin,constraint = computeObjective(w, params)
	print "At end of cuttingPlaneOptimize, objective = " + repr(objective)
	return w
