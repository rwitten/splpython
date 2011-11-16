class Example:
	def __init__(self, inputfile,params,id):
		pass
    
# finds active constraint out of saying that for all \bar y and \bar h
# \psi_i >= \Delta(y_i, \bar y) + w^T \Psi(x_i, \bar y, \bar h) - w^T \Psi(x_i, given_y, given_h)
# or that
# \psi_i >= \Delta(y_i, \bar y) + w^T ( \Psi(x_i, \bar y, \bar h) - \Psi(x_i, given_y, given_h) )
#where vector =( \Psi(x_i, \bar y, \bar h) - \Psi(x_i, given_y, given_h) )
# returns tuple, (const, vector)
def findCuttingPlane(w, params):
	pass

#returns list of maps from classes to scores
def findScoreAllClasses(w, params):
	pass


