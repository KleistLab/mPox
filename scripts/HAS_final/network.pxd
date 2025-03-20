from agent cimport Agent

cdef class Network:
	cdef public dict dict
	cdef public list agents
	cdef public double warm_up, T
	# functions
	cpdef list initialize_agents(self, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] vaccination_vec, long[::1] behaviour_somewhat, long[::1] behaviour_strong)
	cpdef double check_edge(self, int i, int j)
	cpdef double update_edge(self, int i, int j, double t)
	cpdef void update_T(self)
	cpdef list get_R_I_mu(self)