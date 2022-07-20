import numpy as np


class Vector():
	""" Vector Class """
	def __init__(self, R, V, name, color=None, coords=None):
		self.R = R
		self.V = V
		self.name = name
		self.coords = coords
		self.color = color


def isUnitVector(vec):
    ''' Check if a vector is unit '''
    print('Vector NORM: ' + str(np.linalg.norm(vec)))
    return 1 == np.linalg.norm(vec)
