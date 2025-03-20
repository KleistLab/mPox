#cimport agent
#import agent
from agent cimport Agent

from cython cimport view

import numpy as np


cdef class Network:
	def __init__(self, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] vaccination_vec, long[::1] behaviour_somewhat, long[::1] behaviour_strong, double warm_up):
		self.agents = self.initialize_agents(mu_vec, mu_1, lambda_plus_vec, lambda_minus_vec, vaccination_vec, behaviour_somewhat, behaviour_strong)
		self.dict = {}
		self.warm_up = warm_up
		self.T = np.sum(mu_1)/2

	cdef public:
		# init
		cpdef list initialize_agents(self, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] vaccination_vec, long[::1] behaviour_somewhat, long[::1] behaviour_strong):
			cdef int i 
			cdef list res = list()
			for i in range(len(mu_vec)):
				res.append(Agent(i, mu_vec[i], mu_1[i], lambda_plus_vec[i], lambda_minus_vec[i], vaccination_vec[i], behaviour_somewhat[i], behaviour_strong[i]))
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

		cpdef void update_T(self):
			cdef Agent agent
			#cdef double[::1] tmp
			cdef int i

			tmp = np.zeros(len(self.agents))

			for i in range(len(self.agents)):
				tmp[i] = self.agents[i].lambda_plus

			
			self.T = np.sum(tmp)**2- np.sum(tmp**2)
			self.T *= 1/2

		cpdef list get_R_I_mu(self):
			cdef Agent agent
			#cdef double[::1] tmp
			cdef int i

			tmp = np.zeros(len(self.agents))
			tmpR = np.zeros(len(self.agents))
			tmpI = np.zeros(len(self.agents))

			for i in range(len(self.agents)):
				tmp[i] = self.agents[i].lambda_plus
				if self.agents[i].status == "R" or self.agents[i].vaccination_status == 1:
					tmpR[i] = self.agents[i].lambda_plus
				elif self.agents[i].status == "I":
					tmpI[i] = self.agents[i].lambda_plus

			
			lambda_R = np.sum(tmpR)*np.sum(tmp)- np.sum(tmpR**2)
			lambda_I = np.sum(tmpI)*np.sum(tmp)- np.sum(tmpI**2)

			return([lambda_R, lambda_I])

