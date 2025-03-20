from network cimport Network
from pandemic cimport Pandemic

import numpy as np

# random number generator
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int INT_MAX

# logarithm
cdef extern from "math.h":
    double log(double x) nogil


cdef class World:
    def __init__(self, double inf_rate, double diag_rate, double incubation_rate, double recovery_rate, int number_infectious_compartments, double return_vacc, double return_unvacc, int i0, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] behaviour_strong, long[::1] behaviour_somewhat, long[::1] vaccination_vec, long[::1] vaccination_timeline, double behaviour_change, double behaviour_change_rate, double warm_up, int t_max):
        self.network = Network(mu_vec, mu_1, lambda_plus_vec, lambda_minus_vec, vaccination_vec, behaviour_strong, behaviour_somewhat, warm_up)
        self.pandemic = Pandemic(self.network, inf_rate, diag_rate, incubation_rate, recovery_rate, number_infectious_compartments, return_vacc, return_unvacc, i0, behaviour_change, behaviour_change_rate, warm_up)
        self.t = 0
        self.t_max = t_max
        self.steps = 0
        self.delta_t_stored = 0
    
    # simulation
    cpdef void run_world(self):
        cdef double r_0, delta_t, u
        while self.t < self.t_max:
            # sum of propensities
            #r_0 = self.pandemic.r_inf + self.pandemic.r_diag + self.pandemic.r_rec + self.pandemic.r_measure + self.pandemic.r_measure_removal
            #self.pandemic.propensities["behaviour_return"] = 0
            r_0 = self.pandemic.propensities["infection"] + self.pandemic.propensities["diagnosis"] + self.pandemic.propensities["recovery"] + self.pandemic.propensities["behaviour_change"] + self.pandemic.propensities["behaviour_return"]
            #print(self.pandemic.propensities)

            if abs(self.pandemic.propensities["infection"] + self.pandemic.propensities["diagnosis"] + self.pandemic.propensities["recovery"]) < 10**(-6):
                # no infection event possible
                break


            if abs(r_0) < 10**(-6):
                #self.delta_t_stored = delta_t
                break

            # sample next time 
            #delta_t = np.random.exponential(1/r_0)
            if self.delta_t_stored > 0:
                delta_t = self.delta_t_stored
            else:
                u = rand()
                while u == 0:
                    u = rand()
                delta_t = 1/r_0 * log(float(INT_MAX) / rand())

            if self.t + delta_t >= self.t_max:
                self.delta_t_stored = delta_t
                break

            self.delta_t_stored = 0
            self.t += delta_t
            self.step(r_0)

    cpdef void step(self, double r_0):
        cdef double u
        self.steps += 1
        # select next event 
        #u = r_0 * np.random.rand()
        u = r_0 * rand() / float(INT_MAX)

        
        if u < self.pandemic.propensities["infection"]:
            # infection
            self.pandemic.infection(u, self.t)
        elif u <  self.pandemic.propensities["infection"] + self.pandemic.propensities["diagnosis"]:
            # diagnosis
            self.pandemic.diagnosis(u - self.pandemic.propensities["infection"], self.t)
        elif u <  self.pandemic.propensities["infection"] + self.pandemic.propensities["diagnosis"] + self.pandemic.propensities["recovery"]:
            # recovery
            self.pandemic.recovery(u -  self.pandemic.propensities["infection"] - self.pandemic.propensities["diagnosis"], self.t)
        elif u <  self.pandemic.propensities["infection"] + self.pandemic.propensities["diagnosis"] + self.pandemic.propensities["recovery"] + self.pandemic.propensities["behaviour_change"]:
            # random measure
            self.pandemic.measure(u - self.pandemic.propensities["infection"] - self.pandemic.propensities["diagnosis"] - self.pandemic.propensities["recovery"], self.t)
        else:
            self.pandemic.lift_measure(u - self.pandemic.propensities["infection"] - self.pandemic.propensities["diagnosis"] - self.pandemic.propensities["recovery"] - self.pandemic.propensities["behaviour_change"], self.t)




