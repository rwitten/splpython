import copy 
import datetime
import math
import mosek
import numpy
from scipy import sparse
import sys

from imageImplementation import CommonApp

#optimization problem is
# minimize .5 |w|_2^2 + \frac{C}{n} \sum_{i=1}^n \Psi_i
# (which we solve in the one slack formulation


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
	pass


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
	print("response code: %d" % (r))

	task.solutionsummary(mosek.streamtype.msg)

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
	print("mosek says primal is %f and dual is %f" % ( primalObj, dualObj))
	wOut = numpy.mat(xx).T
	return wOut,task.getprimalobj(mosek.soltype.itr), dualObj-primalObj

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
	print("w is" + repr(w.shape) )
	
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
#	task.putintparam(mosek.iparam.intpnt_num_threads ,40)
	task.putintparam(mosek.iparam.sim_max_num_setbacks,	1000)
	# Attach a printer to the task
	task.set_Stream (mosek.streamtype.log, streamprinter)

	task.putmaxnumvar(NUMVAR)
	
	return env, task
def cuttingPlaneOptimize(w, params):
	env,task = initializeMosek(params)
	
	objective,margin, constraint = computeObjective(w, params)
	print "At beginning of cuttingPlaneOptimize, objective = " + repr(objective)
	constraints = []
	margins = []

	notConverged = 1
	LB = - numpy.inf
	UB = numpy.inf

	iter = 1
	while (UB - LB > params.maxDualityGap):
		starttime = datetime.datetime.now()
		
		(xs, garbage) = constraint.nonzero()
		xs = xs.tolist() + [params.lengthW]
		values = constraint.data.tolist() + [1]
		constraints.append( (xs,values)) 
		margins.append(margin)
		startqp = datetime.datetime.now()	
		(w, newLB, dualityGap) = solveQP(constraints, margins, params,env,task)


		
		endqp = datetime.datetime.now()	

		if (newLB > LB) and abs(dualityGap)<=params.maxDualityGap:
			LB = newLB
		
		startFMVC = datetime.datetime.now()	
		(newUB, margin, constraint) = computeObjective(w, params)
		endFMVC= datetime.datetime.now()	
	
		print("returned margin is %d" %( margin - ((w.T * constraint)[0,0])))	
		if margin - ((w.T * constraint)[0,0]) < float(-1e-10):
			print "OUCH: " + repr(margin + ((w.T * constraint)[0,0]))

		assert(margin - ((w.T * constraint)[0,0]) >= float(-1e-10)) 

		if (newUB < UB):
			UB = newUB
		
		endtime= datetime.datetime.now()

		print( "UB is %f and LB is %f on iteration %f" % ( UB, LB,iter) )
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
