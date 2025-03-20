from agent cimport Agent

cdef class Network:
	cdef public dict dict
	cdef public list agents
	cdef public double warm_up
	# functions
	cpdef list initialize_agents(self, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] behaviour_somewhat, long[::1] behaviour_strong, double[::1] inf_rate_vec, double[::1] bc_vec)
	cpdef double check_edge(self, int i, int j)
	cpdef double update_edge(self, int i, int j, double t)
