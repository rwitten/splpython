import copy 
import datetime
import math
import mosek
import numpy
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


def solveDualQP(constraintList, constraints, margins, params, env,task):
	env = mosek.Env ()
	env.set_Stream (mosek.streamtype.log, streamprinter)

	task = env.Task(0,0)
	#task.putintparam(mosek.iparam.sim_max_num_setbacks,	10**8)
	#task.putdouparam(mosek.dparam.intpnt_nl_tol_near_rel, 2)
	#task.putdouparam(mosek.dparam.intpnt_nl_tol_rel_gap, 10**-14)
	#task.putdouparam(mosek.dparam.intpnt_tol_dfeas, 10**-15)
	#task.putdouparam(mosek.dparam.intpnt_tol_step_size, 10**-15)

		

	task.putobjsense(mosek.objsense.minimize)
	constraintsMatrix = sparse.hstack(constraintList)

	FTF = (constraintsMatrix.T*constraintsMatrix).todense()


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

	for i in range(NUMVAR-1): #NO PSI HERE
		for j in range(i+1):
			task.putqobjij(i,j, FTF[i,j])
#	task.putqobjij(NUMVAR-1,NUMVAR-1, .000001) #doesn't matter, makes matrix PD
			
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

	print("FOR DUAL values are primal %f dual %f" %(primalObj,dualObj))

	def objFunc(input):
		x = numpy.asmatrix(input).T
		quadCost = .5*x.T*FTF*x
		penaltyCost = params.C*max(0,numpy.max(numpy.asmatrix(margins).T-FTF*x))
		return (quadCost + penaltyCost)[0,0]

	return wOut,task.getprimalobj(mosek.soltype.itr), primalObj-dualObj

#we want that [w \psi]^t [f_i 1] >= \delta_i
def solveQP(constraints, margins, params,env,task):
	NUMVAR = params.lengthW+1 #remember psi
	NUMCON = len(margins)

	task.append(mosek.accmode.con,1) #1 more constraint
	
	i = NUMCON-1
	task.putbound(mosek.accmode.con,i, mosek.boundkey.lo, margins[i], numpy.inf)
	task.putavec(mosek.accmode.con,i, constraints[i][0],constraints[i][1]) 

	task.putobjsense(mosek.objsense.minimize)
	r=task.optimize()
#	task.solutionsummary(mosek.streamtype.msg)

	xx = numpy.zeros(NUMVAR-1, float)
	task.getsolutionslice(mosek.soltype.itr,
                        mosek.solitem.xx,
                        0,NUMVAR-1, # don't give back psi
                        xx)
	[prosta ,solsta]=task.getsolutionstatus(mosek.soltype.itr)

	yy = numpy.zeros(1, float)
	task.getsolutionslice(mosek.soltype.itr,
                        mosek.solitem.xx,
                        NUMVAR-1,NUMVAR, #we want psi
                        yy)

	primalObj = task.getprimalobj(mosek.soltype.itr)
	dualObj = task.getdualobj(mosek.soltype.itr)

	print("FOR PRIMAL values are primal %f dual %f" % (primalObj,dualObj))

	if abs(dualObj - primalObj)> params.maxDualityGap:
		print("mosek says primal is %f and dual is %f response code %d" % ( primalObj, dualObj, r))

	wOut = numpy.mat(xx).T
	return wOut,task.getprimalobj(mosek.soltype.itr), primalObj-dualObj

def evaluateObjectiveOnPartialQP(w, constraints, margins,params):
	psi = 0
	for i in range(len(margins)):
		psiCon = margins[i]
		for j in range(len(constraints[i][0])-1):
			psiCon = psiCon - w[constraints[i][0][j],0]*constraints[i][1][j]

		if psiCon>psi:
			psi = psiCon 

	return 0.5 * (w.T * w)[0,0] + params.C*psi

def dropConstraints(w, constraintList, margins, idle, constraints, task, params):
	hitlist = []
	psiConMax = - numpy.inf
	psiCons = numpy.zeros(len(idle), float)
	numActive = 0
	for i in range(len(idle)):
		psiCons[i] = margins[i] - (w.T * constraintList[i])[0, 0]
		if psiCons[i] > psiConMax:
			psiConMax = psiCons[i]

	for i in range(len(idle)):
		if psiConMax - psiCons[i] >= params.maxPsiGap * params.C: #My reasoning is that as C gets bigger, w gets bigger, and as w gets bigger, the gaps get bigger
			idle[i] += 1
			if idle[i] >= params.maxTimeIdle:
				hitlist.append(i)

		else:
			numActive += 1
			idle[i] = 0
	
	task.remove(mosek.accmode.con, hitlist)
	hitlist.reverse()
	for j in range(len(hitlist)):
		i = hitlist[j]
		del idle[i]
		del margins[i]
		del constraintList[i]
		del constraints[i]

	print("Dropped %d constraints leaving %d active out of %d total\n"%(len(hitlist), numActive, len(constraints)))

def computeObjective(w, params):
	objective = 0.5 * (w.T * w)[0,0]
	#print("w is" + repr(w.shape) )
	
	(margin, constraint) = CommonApp.findCuttingPlane(w, params)
	objective += params.C*(margin - ((w.T * constraint)[0,0]))
	return (objective, margin, constraint)


def initializeMosek(params):
	env = mosek.Env ()
	task = env.Task(0,0)
	NUMVAR = params.lengthW+1
	task.append(mosek.accmode.var,NUMVAR)
	for j in range(NUMVAR-1):
		task.putbound(mosek.accmode.var,j,mosek.boundkey.ra,-1e4,1e4)
		task.putcj(j, 0)
	qsubi = range(0, NUMVAR-1)
	qsubj = range(0, NUMVAR-1)
	qval = [1]* (NUMVAR-1)
	task.putbound(mosek.accmode.var,NUMVAR-1,mosek.boundkey.lo,0,1e4)
	task.putcj(NUMVAR-1,params.C)

	task.putqobj(qsubi, qsubj, qval)
	
	# Attach a printer to the environment
	env.set_Stream (mosek.streamtype.log, streamprinter)

	# Tricks for the task
#	task.putintparam(mosek.iparam.data_check,mosek.onoffkey.on)
	import multiprocessing
	task.putintparam(mosek.iparam.intpnt_num_threads ,multiprocessing.cpu_count())
	task.putintparam(mosek.iparam.sim_max_num_setbacks,	1000)
	# Attach a printer to the task
	task.set_Stream (mosek.streamtype.log, streamprinter)

	task.putmaxnumvar(NUMVAR)
	
#	task.putmaxnumcon(params.estimatedNumConstraints)

#	task.putmaxnumanz(NUMVAR * params.estimatedNumConstraints)

	return env, task

def cuttingPlaneOptimize(w, params, outerIter):
	env,task = initializeMosek(params)

	print("computing first cutting plane")	
	objective,margin, constraint = computeObjective(w, params)
	print("At beginning of iteration %f, objective = %f" % ( outerIter,objective) ) 
	constraints = None
	idle = []
	margins = []

	notConverged = 1
	LB = - numpy.inf
	UB = numpy.inf

	iter = 1
	while (UB - LB > params.maxDualityGap):
		sys.stdout.flush()

		starttime = datetime.datetime.now()
		idle.append(0)
		constraintList.append(constraint)
		(xs, garbage) = constraint.nonzero()
		xs = xs.tolist() + [params.lengthW]
		values = constraint.data.tolist() + [1]
		constraints.append( (xs,values)) 
		margins.append(margin)
		startqp = datetime.datetime.now()	
		(w, newLB, dualityGap) = solveDualQP(constraintList,constraints, margins, params,env,task)

		(wPrimal, newLBPrimal, dualityGapPrimal) = solveQP(constraints, margins, params,env,task)

		print("distance is %f" % numpy.linalg.norm(w-wPrimal))
#		constraintMatrix = sparse.hstack(constraintList)
#		dualScores = constraintMatrix.T * w
#		primalScores= constraintMatrix.T * wPrimal
	
		endqp = datetime.datetime.now()	

		#dropConstraints(w, constraintList, margins, idle, constraints, task, params)

		if (newLB > LB) and abs(dualityGap)<=params.maxDualityGap:
			LB = newLB
		
		startFMVC = datetime.datetime.now()	
		(newUB, margin, constraint) = computeObjective(w, params)
		endFMVC= datetime.datetime.now()	
	
		if margin - ((w.T * constraint)[0,0]) < float(-1e-10):
			print "OUCH: " + repr(margin + ((w.T * constraint)[0,0]))

		assert(margin - ((w.T * constraint)[0,0]) >= float(-1e-10)) 

		if (newUB < UB):
			UB = newUB
		
		endtime= datetime.datetime.now()

		print( "UB is %f and LB is %f on iteration %f" % ( UB, LB,iter) )
#		print( "New stab at UB was %f" % newUB)
		print( "Total took %f sec, QP took %f sec and FMVC took %f sec" % ( (endtime-starttime).total_seconds(), (endqp-startqp).total_seconds(), (endFMVC-startFMVC).total_seconds()))
		iter+=1
		sys.stdout.flush()

#		import gc
#		startgc = datetime.datetime.now()	
#		gc.collect()
#		endgc= datetime.datetime.now()
#		import pdb; pdb.set_trace()
#		print("gc collect took %f" % (endgc-startgc).total_seconds())

	objective,margin,constraint = computeObjective(w, params)
	print "At end of cuttingPlaneOptimize, objective = " + repr(objective)
	return w
