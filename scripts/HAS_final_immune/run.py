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
second_dose = diseases_parameters.iloc[0]["second_dose"]

population = pd.read_csv(path_to_parameters + "/population.csv")
mu_vec = population["SEX_PARTNERS_MALE_ANAL_UNCODED"].to_numpy(dtype=int)
mu_1 = population["mu_1"].to_numpy()
vaccination_vec = population["MPX_VACC_STATUS_1"].to_numpy(dtype=int)
behaviour_strong_og = population["MPX_VERHALTEN_STRONGLY_REDUCED"].to_numpy(dtype=int)
behaviour_somewhat_og = population["MPX_VERHALTEN_SOMEWHAT_REDUCED"].to_numpy(dtype=int)
lambda_plus_vec = population["lambda_plus"].to_numpy()
lambda_minus_vec = population["lambda_minus"].to_numpy()

parameters = pd.read_csv(path_to_parameters+"/sampled_parameters.csv")

agent_info = pd.read_csv(path_to_parameters+"/has_result_agent_info_"+str(int(parameters.iloc[index_start]["index"])))#+".csv")


N = len(population)

values = np.unique(mu_vec)

df_D_cum = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'vaccine_efficacy1', 'vaccine_efficacy2', 'infection_efficacy', 'replacement_percentage', 'permutation_percentage', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'])
df_I_cum = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'vaccine_efficacy1', 'vaccine_efficacy2', 'infection_efficacy', 'replacement_percentage', 'permutation_percentage', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'])
df_R0 = pd.DataFrame(columns=['inf_prob', 'diag_prob', 'i0', 'vaccine_efficacy1', 'vaccine_efficacy2', 'infection_efficacy', 'replacement_percentage', 'permutation_percentage', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'])

k = 0


# set status of immunized population
agent_info["status"] = "S"
# account for agents who received vaccination after end of first simulation
#agent_info.loc[vaccination_vec==1,"status"] = "V1"
agent_info.loc[agent_info["vaccination_status"]==1, 'status'] = "V1"

# distribute second dose
indices = agent_info.loc[agent_info["vaccination_status"]==1].index
v2_indices = np.random.choice(indices, int(second_dose), replace=False)
agent_info.loc[v2_indices, "status"] = "V2"

# Remove bc from vaccinated and diagnosed agents
agent_info.loc[indices, "b_c"] = 0
behaviour_strong_og[indices] = 0
behaviour_somewhat_og[indices] = 0

agent_info.loc[agent_info["diagnosis_time"] > 0, "b_c"] = 0
behaviour_strong_og[agent_info["diagnosis_time"] > 0] = 0
behaviour_somewhat_og[agent_info["diagnosis_time"] > 0] = 0

# biggest protection comes from infection
if int(sys.argv[4]) == 1:
	agent_info.loc[agent_info["infection_source"]!=0,"status"] = "R"


# get behaviour change from 2022 simulation
bc_vec_og = agent_info["b_c"].to_numpy()

#status_vec = agent_info["status"].tolist()

for l in range(index_end - index_start):
	print("\n Parameter number: "+str(index_start + l))
	params = parameters.iloc[index_start + l]
	index = params.iloc[0]
	inf_prob = params.iloc[1]
	diag_prob = params.iloc[2]
	i0 = int(params.iloc[3])
	behaviour_change = params.iloc[4]
	behaviour_change_rate = params.iloc[5]
	first_outbreak_size = params.iloc[6]
	vaccine_efficacy1 = params.iloc[7]
	vaccine_efficacy2 = params.iloc[8]
	infection_efficacy = params.iloc[9]


	# transform to inf rate
	inf_rate = 1#inf_prob/(1-inf_prob)
	diag_rate = opt.brentq(lambda x: opt_func_diag(x, recovery_rate) - diag_prob, 0, 100)


	# build efficacy vector
	inf_rate_vec = np.ones(N) * inf_prob/(1-inf_prob)
	inf_rate_vec[agent_info["status"] == "V1"] = (1 - vaccine_efficacy1) * inf_prob/(1- (1 - vaccine_efficacy1) * inf_prob)
	inf_rate_vec[agent_info["status"] == "V2"] = (1 - vaccine_efficacy2) * inf_prob/(1- (1 - vaccine_efficacy2) * inf_prob)
	inf_rate_vec[agent_info["status"] == "R"] = (1 - infection_efficacy) * inf_prob/(1- (1 - infection_efficacy) * inf_prob)

	# make copies of vectors that get changed in each iteration
	bc_vec = np.copy(bc_vec_og)
	behaviour_strong = np.copy(behaviour_strong_og)
	behaviour_somewhat = np.copy(behaviour_somewhat_og)

	# replace agents with naive ones
	try:
		replacement_percentage = params.iloc[10]
		# sample replacement for every individual
		replacement_indices = np.random.binomial(1, replacement_percentage, N)
		inf_rate_vec[replacement_indices == 1] = inf_prob/(1-inf_prob)

		if int(sys.argv[5]) == 1:
			# new agents will not change behaviour
			behaviour_strong[replacement_indices == 1] = 0
			behaviour_somewhat[replacement_indices == 1] = 0
			bc_vec[replacement_indices == 1] = 0

	except:
		replacement_percentage = 0
	

	# shuffle efficacy 
	mu_vec = population["SEX_PARTNERS_MALE_ANAL_UNCODED"].to_numpy(dtype=int)
	mu_1 = population["mu_1"].to_numpy()
	lambda_plus_vec = population["lambda_plus"].to_numpy()
	try:
		permutation_percentage = params.iloc[11]
		permutation_indices = np.random.binomial(1, permutation_percentage, N)
		p = np.arange(N)
		p[permutation_indices==1] = np.random.permutation(p[permutation_indices==1])
		mu_vec = mu_vec[p]
		mu_1 = mu_1[p]
		lambda_plus_vec = lambda_plus_vec[p]
	except:
		permutation_percentage = 0


	np.random.seed(l)
	

	# run sim																																
	res_I_cum ,res_D_cum, res_R0 = main_extended(inf_rate, diag_rate, incubation_rate, recovery_rate, number_infectious_compartments, return_vacc, return_unvacc, i0, mu_vec, mu_1, lambda_plus_vec, lambda_minus_vec, behaviour_strong, behaviour_somewhat, behaviour_change, behaviour_change_rate, inf_rate_vec, bc_vec, warm_up, t_max)
	
	tmp = np.diff(np.asarray(res_D_cum))
	res = np.asarray(res_D_cum)
	res[1:] = tmp

		
	# prepare I_cum
	res2 = np.asarray(res_I_cum)
	res2[1:] = np.diff(np.asarray(res_I_cum))

	res3 = np.asarray(res_R0)
	

	# save the result
	df_D_cum.loc[k] = [inf_prob, diag_prob, i0, vaccine_efficacy1, vaccine_efficacy2, infection_efficacy, replacement_percentage, permutation_percentage] + list(res) 
	df_I_cum.loc[k] = [inf_prob, diag_prob, i0, vaccine_efficacy1, vaccine_efficacy2, infection_efficacy, replacement_percentage, permutation_percentage] + list(res2)
	df_R0.loc[k] = [inf_prob, diag_prob, i0, vaccine_efficacy1, vaccine_efficacy2, infection_efficacy, replacement_percentage, permutation_percentage] + list(res3)
	k += 1
	

df_D_cum.to_csv(path_to_results+"/has_result_D_cum_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_I_cum.to_csv(path_to_results+"/has_result_I_cum_"+str(index_start)+"_"+str(index_end)+".csv", index=False)
df_R0.to_csv(path_to_results+"/has_result_R0_"+str(index_start)+"_"+str(index_end)+".csv", index=False)

print("--- %s seconds ---" % (time.time() - start_time))
#print(process.memory_info().rss)