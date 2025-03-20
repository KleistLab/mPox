import numpy as np

from agent cimport Agent
from network cimport Network

cdef class Pandemic:
	cdef public Network network
	cdef public double  warm_up, inf_rate, diag_rate, incubation_rate, recovery_rate, return_vacc, return_unvacc, behaviour_change, behaviour_change_rate, vaccine_efficacy 
	cdef public list I_list, S_list, D_list, R_list, V_list, active_measure_list, I_degree_list, possible_measure_list
	cdef public int I_cum, D_cum, measure_threshold, M_count, total_immunized, number_infectious_compartments
	cdef public dict propensities
	# functions
	cpdef list initialize_sus(self)
	cpdef list initialize_inf(self, i0)
	cpdef list initialize_vaccs(self)
	cpdef void infection(self, double u, double t)
	cpdef void diagnosis(self, double u, double t)
	cpdef void recovery(self, double u, double t)
	cpdef void measure(self, double u, double t)
	cpdef vaccination(self, double number)
	cpdef void lift_measure(self, double u, double t)
	cpdef double[:,:] create_dataframe(self)
