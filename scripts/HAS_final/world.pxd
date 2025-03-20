from network cimport Network
from pandemic cimport Pandemic

import numpy as np

cdef class World:
    cdef public Network network
    cdef public Pandemic pandemic
    cdef public double t, t_max, delta_t_stored
    cdef public int steps
    # functions
    cpdef void run_world(self)
    cpdef void step(self, double r_0)