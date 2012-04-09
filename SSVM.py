import copy 
import datetime
import logging
import math
import mosek
import numpy
from scipy import optimize
from scipy import sparse
import sys

import Params
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

# solves "Cutting-Plane Training of Structural SVMs", Optimization Problem 6
def solveDualQPV2(FTF, constraintsMatrix, margins, params, env,task):
	def evalObjective(arg):
		arg = numpy.asmatrix(arg).T
		cost = (.5 * arg.T* FTF * arg)[0,0]
		cost += max(0,params.C* numpy.max(numpy.asmatrix(margins).T - FTF* arg ))
		return cost

	NUMVAR = FTF.shape[1]
	NUMCON = 1

	if task.NUMVAR<NUMVAR:
		task.append(mosek.accmode.var,NUMVAR-task.NUMVAR)
		task.NUMVAR=NUMVAR

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


	return wOut,task.getprimalobj(mosek.soltype.itr), primalObj-dualObj

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
	(margin, constraint,lvs) = CommonApp.findCuttingPlane(w, params)
	objective += params.C*(margin - ((w.T * constraint)[0,0]))
	return (objective, margin, constraint,lvs)

def initializeMosek(params):
	env = mosek.Env ()
	task = env.Task(0,0)
	task.NUMVAR=0
	task.putobjsense(mosek.objsense.maximize)
	task.append(mosek.accmode.con,1) #one constraint at all times
	return env, task

def cuttingPlaneOptimize(w, params, outerIter):
	env,task = initializeMosek(params)

	objective,margin, constraint,lv = computeObjective(w, params)
	logging.debug("At beginning of iteration %f, objective = %f" % ( outerIter,objective) ) 
	lvsList = [lv]
	F = constraint 
	FTF = (F.T*F).todense()
	
	margins = [margin]

	notConverged = 1
	LB = - numpy.inf
	UB = numpy.inf

	iter = 1
	while (UB - LB > params.maxDualityGap):
		logging.debug("Starting QP solve + constraint add")

		starttime = datetime.datetime.now()
		startqp = datetime.datetime.now()

		(w, newLB, dualityGap) = solveDualQPV2(FTF, F, margins, params,env,task)
		logging.debug("Done with QP solve")

		endqp = datetime.datetime.now()	

		if (newLB > LB) and abs(dualityGap)<=params.maxDualityGap:
			LB = newLB
		
		startFMVC = datetime.datetime.now()	
		newUB, margin, constraint,lv = computeObjective(w, params)
		endFMVC= datetime.datetime.now()	

		assert(margin - ((w.T * constraint)[0,0]) >= float(-1e-10)) 

		if (newUB < UB):
			UB = newUB

		margins.append(margin)
		newPortion = (F.T*constraint).todense()
		FTF = numpy.hstack( [FTF, newPortion])
		newPortionPadded = numpy.hstack( [ newPortion.T, (constraint.T*constraint).todense()])
		FTF = numpy.vstack( [FTF, newPortionPadded])
	
		F=sparse.hstack( [F, constraint])
		lvsList.append(lv)
		endtime= datetime.datetime.now()

		logging.debug( "UB is %f and LB is %f on iteration %f" % ( UB, LB,iter) )
		logging.debug( "TIMING step %f sec, QP took %f sec and FMVC took %f sec" % ( (endtime-starttime).total_seconds(), (endqp-startqp).total_seconds(), (endFMVC-startFMVC).total_seconds()))
		iter+=1

	objective,margin,constraint,latestLVs = computeObjective(w, params)
	logging.debug("At end of cuttingPlaneOptimize, objective = " + repr(objective))

	optState = Params.Params()
	optState.F = F.todense()
	optState.margins = margins
	optState.latents = lvsList
	optState.w=w
	optState.outerIter= outerIter
	return w, optState
