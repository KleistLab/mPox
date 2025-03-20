#cimport agent
#import agent
from agent cimport Agent

from cython cimport view

import numpy as np


cdef class Network:
	def __init__(self, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] behaviour_somewhat, long[::1] behaviour_strong, double[::1] inf_rate_vec, double[::1] bc_vec, double warm_up):
		self.agents = self.initialize_agents(mu_vec, mu_1, lambda_plus_vec, lambda_minus_vec, behaviour_somewhat, behaviour_strong, inf_rate_vec, bc_vec)
		self.dict = {}
		self.warm_up = warm_up

	cdef public:
		# init
		cpdef list initialize_agents(self, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] behaviour_somewhat, long[::1] behaviour_strong, double[::1] inf_rate_vec, double[::1] bc_vec):
			cdef int i 
			cdef list res = list()
			for i in range(len(mu_vec)):
				res.append(Agent(i, mu_vec[i], mu_1[i], lambda_plus_vec[i], lambda_minus_vec[i], behaviour_somewhat[i], behaviour_strong[i], inf_rate_vec[i], bc_vec[i]))
			return(res)

		# contact updates
		cpdef double check_edge(self, int i, int j):
			cdef str identifier = str(min(i, j)) + "_" + str(max(i, j))
			try:
				return(self.dict[identifier])
			except:
				return(self.warm_up)

				
		cpdef double update_edge(self, int i, int j, double t):
			cdef str identifier = str(min(i, j)) + "_" + str(max(i, j))
			self.dict[identifier] = t