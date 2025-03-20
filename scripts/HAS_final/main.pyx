from world cimport World

import numpy as np
import sys

def main_extended(double inf_rate, double diag_rate, double incubation_rate, double recovery_rate, int number_infectious_compartments, double return_vacc, double return_unvacc, int i0, long[::1] mu_vec, double[::1] mu_1, double[::1] lambda_plus_vec, double[::1] lambda_minus_vec, long[::1] behaviour_strong, long[::1] behaviour_somewhat, long[::1] vaccination_vec, long[::1] vaccination_timeline, double behaviour_change, double behaviour_change_rate, double warm_up, int t_max):
	cdef double[::1] res_I, res_D, res_S, res_R, res_lambda_plus, res_I_cum, res_D_cum, res_I_cum_high, res_M_count, res_lambda_plus_R, res_mean_degree_I, tmp
	cdef double[:,:] agent_info
	cdef int i, t, l
	cdef World world
	cdef double tmp3
	cdef list tmp2

	res_S = np.zeros((t_max + 1))
	res_I = np.zeros((t_max + 1))
	res_D = np.zeros((t_max + 1))
	res_R = np.zeros((t_max + 1))
	res_I_cum = np.zeros((t_max + 1))
	res_D_cum = np.zeros((t_max + 1))
	res_lambda_plus = np.zeros((t_max + 1))
	res_M_count = np.zeros((t_max + 1))
	res_lambda_plus_R = np.zeros((t_max + 1))
	res_mean_degree_I = np.zeros((t_max + 1))
	#res_inf_prob = np.zeros((len(lambda_plus_vec), t_max + 1))
	tmp = np.zeros(len(lambda_plus_vec))


	#world = pickle.load(filehandler)
	world = World(inf_rate, diag_rate, incubation_rate, recovery_rate, number_infectious_compartments, return_vacc, return_unvacc, i0, mu_vec, mu_1, lambda_plus_vec, lambda_minus_vec, behaviour_strong, behaviour_somewhat, vaccination_vec, vaccination_timeline, behaviour_change, behaviour_change_rate, warm_up, 1)
		
	# first step
	res_S[0] = len(world.pandemic.S_list)
	res_I[0] = len(world.pandemic.I_list)
	res_D[0] = len(world.pandemic.D_list)
	res_R[0] = len(world.pandemic.R_list)
	res_I_cum[0] = world.pandemic.I_cum
	res_D_cum[0] = world.pandemic.D_cum
	res_M_count[0] = world.pandemic.M_count
	res_lambda_plus[0] = world.network.T

	tmp3 = 0
	for agent in world.pandemic.I_list:
		tmp3 += agent.mu_1
	res_mean_degree_I[0] = tmp3/len(world.pandemic.I_list)



	#for l in range(len(lambda_plus_vec)):
	#	res_inf_prob[l,0] = 0 if status_vec[l] == "S" else 1

	t = 1
	while t <= t_max:
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*int((t)*20/t_max), (t)*100/t_max))
		#sys.stdout.write(str(i)+"/"+str(sims))
		sys.stdout.flush()

		# introduce vaccines
		world.pandemic.vaccination(vaccination_timeline[t])
		if vaccination_vec[t] > 0:
			world.delta_t_stored = 0

		# execute step
		world.t_max = t
		world.run_world()
		world.network.update_T()

		res_S[t] = len(world.pandemic.S_list)
		res_I[t] = len(world.pandemic.I_list)
		res_D[t] = len(world.pandemic.D_list)
		res_R[t] = len(world.pandemic.R_list)
		res_I_cum[t] = world.pandemic.I_cum
		res_D_cum[t] = world.pandemic.D_cum
		res_M_count[t] = world.pandemic.M_count

		tmp2 = world.network.get_R_I_mu()

		res_lambda_plus[t] = world.network.T
		res_lambda_plus_R[t] = tmp2[0]
		if len(world.pandemic.I_list) == 0:
			res_mean_degree_I[t] = 0
		else:
			res_mean_degree_I[t] = tmp2[1]/len(world.pandemic.I_list)

		t += 1

	agent_info = world.pandemic.create_dataframe()

		
	return(res_S, res_I, res_D, res_R, res_I_cum, res_D_cum, res_lambda_plus, res_M_count, res_lambda_plus_R, world.pandemic.total_immunized, res_mean_degree_I, world.pandemic.I_degree_list, agent_info)




