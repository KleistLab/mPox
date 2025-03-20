from world cimport World

import numpy as np
import sys

def main_extended(double inf_rate, double diag_rate, double incubation_rate, double recovery_rate, int number_infectious_compartments, double return_vacc, double return_unvacc, int i0, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] behaviour_strong, long[::1] behaviour_somewhat, double behaviour_change, double behaviour_change_rate, double[::1] inf_rate_vec, double[::1] bc_vec, double warm_up, int t_max):
	cdef double[::1] res_I_cum, res_D_cum, res_R0
	cdef int t
	cdef World world

	res_I_cum = np.zeros((t_max + 1))
	res_D_cum = np.zeros((t_max + 1))
	res_R0 = np.zeros((t_max + 1))
	
	world = World(inf_rate, diag_rate, incubation_rate, recovery_rate, number_infectious_compartments, return_vacc, return_unvacc, i0, mu_vec, mu_1, lambda_plus_vec, lambda_minus_vec, behaviour_strong, behaviour_somewhat, behaviour_change, behaviour_change_rate, inf_rate_vec, bc_vec, warm_up, 1)
		
	# first step
	res_I_cum[0] = world.pandemic.I_cum
	res_D_cum[0] = world.pandemic.D_cum
	res_R0[0] = world.pandemic.propensities["infection"]/len(world.pandemic.I_list)

	#for l in range(len(lambda_plus_vec)):
	#	res_inf_prob[l,0] = 0 if status_vec[l] == "S" else 1

	t = 1
	while t <= t_max:
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*int((t)*20/t_max), (t)*100/t_max))
		#sys.stdout.write(str(i)+"/"+str(sims))
		sys.stdout.flush()

		# execute step
		world.t_max = t
		world.run_world()

		res_I_cum[t] = world.pandemic.I_cum
		res_D_cum[t] = world.pandemic.D_cum
		if len(world.pandemic.I_list) > 0:
			res_R0[t] = world.pandemic.propensities["infection"]/len(world.pandemic.I_list)
		else:
			res_R0[t] = 0

		t += 1


		
	return(res_I_cum, res_D_cum, res_R0)