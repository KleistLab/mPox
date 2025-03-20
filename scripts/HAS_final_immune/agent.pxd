cdef class Agent:
	cdef public int id, I_round, infectious
	cdef public str status
	cdef public dict parameters 
	cdef public double mu_1, infection_risk, diagnosis_risk, recover_risk, lambda_plus, lambda_minus, inf_rate, infection_time, behaviour_change_time, behaviour_change_rate, behaviour_return, behaviour_return_time, b_c, exposure_time, diagnosis_time, recovery_time
	# functions
	cpdef void get_infection_risk(self, list I_list)
	cpdef Agent select_infected_partner(self, list I_list, double u)
	cpdef void update_risk_added_I(self, Agent agent)
	cpdef void update_risk_added_R(self, Agent agent)
	cpdef void update_risk_added_M(self, Agent agent, double beta)
	cpdef void become_infected(self, double incubation_rate)
	cpdef void become_infectious(self, double t, double recover_rate)
	cpdef void become_diagnosed(self)
	cpdef void become_recovered(self)
	cpdef void update_risk_removed_M(self, Agent agent)