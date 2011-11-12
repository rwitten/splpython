import datetime
import mosek
import numpy
from scipy import sparse

from imageImplementation import ImageApp as App
#optimization problem is
# minimize .5 |w|_2^2 + \frac{C}{n} \sum_{i=1}^n \Psi_i
# (which we solve in the one slack formulation


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
	pass


#we want that [w \psi]^t [f_i 1] >= \delta_i
def solveQP(constraints, margins, params):
	NUMVAR = params.lengthW+1 #remember psi
	NUMCON = len(margins)

	# Make a MOSEK environment
	env = mosek.Env ()
	# Attach a printer to the environment
	env.set_Stream (mosek.streamtype.log, streamprinter)

	# Create a task
	task = env.Task(0,0)
	# Attach a printer to the task
	task.set_Stream (mosek.streamtype.log, streamprinter)

	task.putmaxnumvar(NUMVAR)
	task.putmaxnumcon(NUMCON)

	task.append(mosek.accmode.con,NUMCON) #NUMCON empty constraints

	task.append(mosek.accmode.var,NUMVAR) #NUMVAR variables fixed at zero
	for j in range(NUMVAR-1):
		#task.putbound(mosek.accmode.var,j,mosek.boundkey.fr,-numpy.inf,numpy.inf)
		task.putbound(mosek.accmode.var,j,mosek.boundkey.ra,-1e4,1e4)
		task.putcj(j, 0)

	task.putbound(mosek.accmode.var,NUMVAR-1,mosek.boundkey.lo,0,1e4)
	task.putcj(NUMVAR-1,1)

	for i in range(NUMCON):
		task.putbound(mosek.accmode.con,i, mosek.boundkey.lo, margins[i], numpy.inf)
		task.putavec(mosek.accmode.con,i, constraints[i][0],constraints[i][1]) 


	qsubi = range(0, NUMVAR-1)	
	qsubj = range(0, NUMVAR-1)
	qval = [1]* (NUMVAR-1)

	task.putqobj(qsubi, qsubj, qval)


	task.putobjsense(mosek.objsense.minimize)
	task.optimize()
	task.solutionsummary(mosek.streamtype.msg)

	xx = numpy.zeros(NUMVAR-1, float)
	task.getsolutionslice(mosek.soltype.itr,
                        mosek.solitem.xx,
                        0,NUMVAR-1, # don't give back psi
                        xx)

	wOut = numpy.mat(xx).T
	return wOut,task.getprimalobj(mosek.soltype.itr)


def computeObjective(w, params):
	objective = 0.5 * (w.T * w)[0,0]
	(margin, constraint) = App.findCuttingPlane(w, params)
	objective += margin - ((w.T * constraint)[0,0])
	return objective

def cuttingPlaneOptimize(w, params):
	objective = computeObjective(w, params)
	print "At beginning of cuttingPlaneOptimize, objective = " + repr(objective)
	constraints = []
	margins = []

	notConverged = 1
	LB = - numpy.inf
	UB = numpy.inf

	(margin, constraint) = App.findCuttingPlane(w, params)

	while (UB - LB > params.maxDualityGap):
		starttime = datetime.datetime.now()
		(xs, garbage) = constraint.nonzero()
		xs = xs.tolist() + [params.totalLength]
		values = constraint.data.tolist() + [1]
		
		constraints.append( (xs,values)) 
		margins.append(margin)

		startqp = datetime.datetime.now()	
		(w, newLB) = solveQP(constraints, margins, params)
		endqp = datetime.datetime.now()	

		if(newLB > LB):
			LB = newLB
		
		startFMVC = datetime.datetime.now()	
		(margin, constraint) = App.findCuttingPlane(w, params)
		endFMVC= datetime.datetime.now()	
		
		assert(margin - ((w.T * constraint)[0,0]) >= float(0.0)) #Even this assertion isn't strong enough - NONE of the vectors that sum up to the cutting plane should get a negative number when dot-producted with w and subtracted from the appropriate delta
		newUB = margin - (w.T * constraint)[0,0] + 0.5 * (w.T * w)[0,0]

		if (newUB < UB):
			UB = newUB
		
		endtime= datetime.datetime.now()

		print( "UB is %f and LB is %f" % ( UB, LB) )
		print( "Total took %f sec, QP took %f sec and FMVC took %f sec" % ( (endtime-starttime).total_seconds(), (endqp-startqp).total_seconds(), (endFMVC-startFMVC).total_seconds()))

	objective = computeObjective(w, params)
	print "At end of cuttingPlaneOptimize, objective = " + repr(objective)
	return w
