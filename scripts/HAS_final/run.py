from scipy.stats import poisson, multinomial
import numpy as np
#import copy
import time
import pandas as pd
import sys
import pickle 
#import psutil
#process = psutil.Process()
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# for D calculation
from scipy.linalg import expm
import scipy.optimize as opt
def opt_func_diag(x, intensity):
    Q = np.array([[-intensity-x,intensity,0,0,0,0,x],[0,-intensity-x,intensity,0,0,0,x], [0,0,-intensity-x,intensity,0,0,x], [0,0,0,-intensity-x,intensity,0,x], [0,0,0,0,-intensity-x,intensity,x], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]])
    x = np.array([1,0,0,0,0,0,0])
    P = np.matmul(x,expm(Q*1000))
    return(P[6])

start_time = time.time()

from main import *


index_start = int(sys.argv[1])
index_end = int(sys.argv[2])
name = sys.argv[3]

path_to_results = ""
t_max = 25
seed = 10
warm_up = -5.0 # can be set to negative value in order to "evolve" network before pandemic hits 

# load parameters from files?
name_param = name

path_to_parameters = ""

diseases_parameters = pd.read_csv(path_to_parameters+"/diseases_parameters.csv")
incubation_rate = diseases_parameters.iloc[0]["incubation_rate"]
recovery_rate = diseases_parameters.iloc[0]["recovery_rate"]
number_infectious_compartments = int(diseases_parameters.iloc[0]["number_infectious_compartments"])
return_vacc = diseases_parameters.iloc[0]["return_vacc"]
return_unvacc = diseases_parameters.iloc[0]["return_unvacc"]

population = pd.read_csv(path_to_parameters + "/population.csv")
mu_vec = population["SEX_PARTNERS_MALE_ANAL_UNCODED"].to_numpy(dtype=int)
mu_1 = population["mu_1"].to_numpy()
vaccination_vec = population["MPX_VACC_STATUS_1"].to_numpy(dtype=int)
behaviour_strong = population["MPX_VERHALTEN_STRONGLY_REDUCED"].to_numpy(dtype=int)
behaviour_somewhat = population["MPX_VERHALTEN_SOMEWHAT_REDUCED"].to_numpy(dtype=int)
lambda_plus_vec = population["lambda_plus"].to_numpy()
lambda_minus_vec = population["lambda_minus"].to_numpy()


parameters = pd.read_csv(path_to_parameters+"/sampled_parameters.csv")
vaccination_timeline = np.load(path_to_parameters+"/vaccination_timeline.npy")
berlin = np.load(path_to_parameters+"/reported_cases.npy")

N = len(population)

values = np.unique(mu_vec)

df_D_cum = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25', 'total_immunized', 'LL'])
df_I_cum = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25', 'total_immunized', 'LL'])
df_I_degree_list = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate'] + values.tolist()  +['total_immunized', 'LL'])
df_lambda = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25', 'total_immunized', 'LL'])
df_M_count = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25', 'total_immunized', 'LL'])
df_R = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25', 'total_immunized', 'LL'])
df_R_lambda = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25', 'total_immunized', 'LL'])
df_mean_degree_I = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'behaviour_change', 'behaviour_change_rate', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25', 'total_immunized', 'LL'])

k = 0


for l in range(index_end - index_start):
	print("\n Parameter number: "+str(index_start + l))
	params = parameters.iloc[index_start + l]
	inf_prob = params.iloc[0]
	diag_prob = params.iloc[1]
	i0 = int(params.iloc[2])
	behaviour_change = params.iloc[3]
	behaviour_change_rate = params.iloc[4]

	print(inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate)

	# transform to inf rate
	inf_rate = inf_prob/(1-inf_prob)
	diag_rate = opt.brentq(lambda x: opt_func_diag(x, recovery_rate) - diag_prob, 0, 100)


	np.random.seed(l)

	# run sim																																
	res_S, res_I, res_D, res_R, res_I_cum ,res_D_cum, res_lambda_plus, res_M_count, res_lambda_plus_R, total_immunized, res_mean_degree_I, res_I_degree_list, agent_info = main_extended(inf_rate, diag_rate, incubation_rate, recovery_rate, number_infectious_compartments, return_vacc, return_unvacc, i0, mu_vec, mu_1, lambda_plus_vec, lambda_minus_vec, behaviour_strong, behaviour_somewhat, vaccination_vec, vaccination_timeline, behaviour_change, behaviour_change_rate, warm_up, t_max)
	
	tmp = np.diff(np.asarray(res_D_cum))
	res = np.asarray(res_D_cum)
	res[1:] = tmp
	res7 = np.asarray(res_mean_degree_I)

	# calculate LL
	LL = 0
	for i in range(20):
		if res[i] != 0:
			LL += np.log(poisson.pmf(berlin[i],res[i]))
		LL += np.log(poisson.pmf(np.sum(berlin[20:]),np.sum(res[20:])))

		

	# prepare I_cum, I_cum_high, M_count
	res2 = np.asarray(res_I_cum)
	res2[1:] = np.diff(np.asarray(res_I_cum))


	res4 = np.asarray(res_M_count)
	res4[1:] = np.diff(np.asarray(res_M_count))

	res5 = np.asarray(res_R)

	res6 = np.asarray(res_lambda_plus_R)


	values_inf, counts = np.unique(np.asarray(res_I_degree_list), return_counts = True)

	res_degree_I = np.zeros(len(values))

	s = 0
	for i in range(len(values)):
		if s == len(values_inf):
			continue
		if values[i] == values_inf[s]:
			res_degree_I[i] = counts[s]
			s += 1 
	

	# save the result
	df_D_cum.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + list(res) + [total_immunized] + [LL]
	df_I_cum.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + list(res2) + [total_immunized] + [LL]
	#df_I_degree_list.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + ['_'.join(str(x) for x in res_I_degree_list)] + [total_immunized] + [LL]
	df_I_degree_list.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + res_degree_I.tolist() + [total_immunized] + [LL]
	df_lambda.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + list(np.asarray(res_lambda_plus)) + [total_immunized] + [LL]
	df_M_count.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + list(res4) + [total_immunized] + [LL]
	df_R.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + list(res5) + [total_immunized] + [LL]
	df_R_lambda.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + list(res6) + [total_immunized] + [LL]
	df_mean_degree_I.loc[k] = [inf_prob, diag_prob, i0, behaviour_change, behaviour_change_rate] + list(res7) + [total_immunized] + [LL]

	if len(sys.argv) == 5:
		if LL > float(sys.argv[4]):	
			agent_info = np.asarray(agent_info)
			df_agent_info = pd.DataFrame(data=agent_info,columns=["behaviour_change_time", "behaviour_return_time", "b_c", "exposure_time", "infection_time", "diagnosis_time", "recovery_time", "vaccination_status", "mu", "infection_source"])
			df_agent_info.to_csv(path_to_results+"/has_result_agent_info_"+str(index_start + l))
	k += 1

df_D_cum.to_csv(path_to_results+"/has_result_D_cum_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_I_cum.to_csv(path_to_results+"/has_result_I_cum_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_I_degree_list.to_csv(path_to_results+"/has_result_I_degree_list_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_lambda.to_csv(path_to_results+"/has_result_lambda_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_M_count.to_csv(path_to_results+"/has_result_M_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_R.to_csv(path_to_results+"/has_result_R_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_R_lambda.to_csv(path_to_results+"/has_result_R_lambda_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_mean_degree_I.to_csv(path_to_results+"/has_result_mean_degree_I_"+str(index_start)+"_"+str(index_end)+".csv", index=False)

print("--- %s seconds ---" % (time.time() - start_time))
#print(process.memory_info().rss)






