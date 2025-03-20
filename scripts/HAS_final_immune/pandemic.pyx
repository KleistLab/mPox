import numpy as np
import random

from libc.math cimport exp 

from cython cimport view

# random number generator
from libc.stdlib cimport rand
cdef extern from "limits.h":
	int INT_MAX

# logarithm
cdef extern from "math.h":
	double log(double x) nogil

# fast sum
cdef double sum_tmv(double[::1] arr):
	cdef size_t i, I
	cdef double total = 0
	I = arr.shape[0]
	for i in range(I):
		total += arr[i]
	return total

from agent cimport Agent
from network cimport Network


cdef class Pandemic:
	def __init__(self, Network network, double inf_rate, double diag_rate, double incubation_rate, double recovery_rate, int number_infectious_compartments, double return_vacc, double return_unvacc, int i0, double behaviour_change, double behaviour_change_rate, double warm_up):
		self.network = network
		self.inf_rate = inf_rate
		self.diag_rate = diag_rate
		self.incubation_rate = incubation_rate
		self.recovery_rate = recovery_rate
		self.number_infectious_compartments = number_infectious_compartments
		self.return_vacc = return_vacc
		self.return_unvacc = return_unvacc
		self.behaviour_change = behaviour_change
		self.behaviour_change_rate = behaviour_change_rate
		self.warm_up = warm_up
		self.I_cum = 0
		self.D_cum = 0
		self.total_immunized = 0
		self.I_degree_list = []
		self.propensities = {'infection': 0, 'diagnosis': 0, 'recovery': 0, 'behaviour_change': 0, 'behaviour_return': 0}
		self.possible_measure_list = list()
		self.I_list = self.initialize_inf(i0)
		self.S_list = self.initialize_sus()
		self.D_list = list()
		self.R_list = list()
		self.active_measure_list = list()


	cdef public:
		# init 
		# TOUCHED
		cpdef list initialize_sus(self):
			cdef list res = list()
			cdef Agent agent
			cdef int i = 0
			for agent in self.network.agents:
				if agent in self.I_list:
					continue
				agent.get_infection_risk(self.I_list)
				self.propensities["infection"] += self.inf_rate * agent.infection_risk
				res.append(agent)
				#if agent.behaviour_change_rate > 0:
				#	self.possible_measure_list.append(agent)
				#	self.propensities["behaviour_change"] += agent.behaviour_change_rate * self.behaviour_change_rate
			return(res)

		# TOUCHED
		cpdef list initialize_inf(self, i0):
			cdef list res = list()
			cdef Agent agent
			cdef int id_I

			#for i in I_ids:
			while i0 - self.I_cum > 0:
				# select agent
				id_I = np.random.choice(len(self.network.agents), 1)
				agent = self.network.agents[id_I]
				# check if agent is already infected or has less than 5 expected partners
				if agent.status == "I" or agent.parameters["mu_i"] < 5:
					continue
				# agent is eligable, perform infection
				# initially infected agents are already infectious (no incubation period)
				agent.become_infected(0)
				agent.become_infectious(self.warm_up, self.recovery_rate)
				self.propensities["diagnosis"] += self.diag_rate
				self.propensities["recovery"] += agent.recover_risk
				res.append(agent)
				self.I_cum += 1
				self.I_degree_list.append(int(agent.parameters["mu_i"]))
				#if agent.behaviour_strong > 0:
				#	self.possible_measure_list.append(agent)
			return(res)



		# pandemic actions
		cpdef void infection(self, double u, double t):
			cdef double tmp_sum, delta_t, lambda_exp, c1, edge_prob_t,
			cdef int id_S
			cdef Agent agent_I, agent_S

			# select susceptible agent
			tmp_sum = 0
			id_S = 0
			u *= 1/self.inf_rate
			if len(self.S_list) == 0:
				return
			while tmp_sum < u:
				tmp_sum += self.S_list[id_S].infection_risk
				id_S += 1

			tmp_sum -= self.S_list[id_S-1].infection_risk
			agent_S = self.S_list[id_S - 1]

			# select infected agent
			u -= tmp_sum
			if len(self.I_list) == 0:
				return
			agent_I = agent_S.select_infected_partner(self.I_list, u)

			# check if edge between both agents still exists:
			lambda_exp = agent_I.lambda_plus*agent_S.lambda_plus + agent_I.lambda_minus*agent_S.lambda_minus
			c1 = agent_I.lambda_plus*agent_S.lambda_plus/(lambda_exp)

			# get time between last observation
			last_t = self.network.check_edge(agent_I.id, agent_S.id)

			delta_t = t - last_t
			# sample leaped rejection event 
			e_x = agent_S.inf_rate*(1 - c1)
			#delta_tx = np.random.exponential(1/e_x)
			delta_tx = 1/e_x * log(float(INT_MAX) / rand())
			# there was no leaped rejection step if the infection just happend 
			if t - agent_I.infection_time < delta_tx:
				delta_tx = delta_t

			edge_prob_t = 1 - exp(-lambda_exp * min(delta_t, delta_tx) )

			#if edge_prob_t >= np.random.rand(): 
			if edge_prob_t >= rand() / float(INT_MAX):
				# infect agent_S
				# agent_S starts incubation period

				# remove infection risk and infect agent
				self.propensities["infection"] -= agent_S.infection_risk * self.inf_rate
				agent_S.become_infected(self.incubation_rate)
				agent_S.exposure_time = t

				# change lists
				self.I_list.append(agent_S)
				self.S_list.pop(id_S - 1)
				self.I_cum += 1
				self.I_degree_list.append(int(agent_S.parameters["mu_i"]))
				
				# update rec propensities (= symptom onset)
				# fixed to be 1 in this step
				#self.r_rec += agent_S.parameters["rec_i"]
				self.propensities["recovery"] += agent_S.recover_risk

				# decrease measure rate
				#if self.r_measure > 0:
				#	self.r_measure -= agent_S.measure_rate
				#agent_S.measure_rate = 0	
				if agent_S.behaviour_change_rate > 0 and self.propensities["behaviour_change"]>0:
					self.possible_measure_list.pop(self.possible_measure_list.index(agent_S))
					#if self.r_measure > 0:
					self.propensities["behaviour_change"] -= agent_S.behaviour_change_rate * self.behaviour_change_rate
					#self.r_measure -= agent_S.measure_rate

			else:
				# update edge dict
				self.network.update_edge(agent_I.id, agent_S.id, t)


		# treat diagnosis like recovery (absorbing state)
		cpdef void diagnosis(self, double u, double t):
			cdef double tmp
			cdef int id_I 
			cdef Agent agent_I, agent

			# select infected agent to receive the diagnosis
			tmp_sum = 0
			id_I = 0
			while tmp_sum < u:
				agent_I = self.I_list[id_I]
				tmp_sum += self.diag_rate * agent_I.infectious #self.I_list[id_I].diagnosis_risk
				id_I += 1

			#agent_I = self.I_list[id_I - 1]

			# remove risk of infection from agent_I
			if agent_I.infectious == 1:
				self.propensities["infection"] = 0
				for agent in self.S_list:
					agent.update_risk_added_R(agent_I)
					self.propensities["infection"] += agent.infection_risk * self.inf_rate

			# change lists
			self.D_list.append(agent_I)
			self.I_list.pop(id_I-1)
			self.D_cum += 1

			if agent_I.lambda_plus < agent_I.parameters["lambda_plus"]:
				self.active_measure_list.pop(self.active_measure_list.index(agent_I))
				self.propensities["behaviour_return"] -= agent_I.behaviour_return
				agent_I.behaviour_return = 0


			# update diagnosis risk
			self.propensities["diagnosis"] -= self.diag_rate

			# receive diagnosis
			agent_I.become_diagnosed()
			agent_I.diagnosis_time = t

			# increase behaviour change rate
			if self.D_cum > 1 and self.behaviour_change_rate > 0:
				self.propensities["behaviour_change"] *= 1/self.behaviour_change_rate
				self.behaviour_change_rate *= self.D_cum/(self.D_cum-1)
				self.propensities["behaviour_change"] *= self.behaviour_change_rate

			if self.D_cum == 20:
				# activate measures
				for agent in self.S_list:
					if agent.behaviour_change_rate > 0:
						self.possible_measure_list.append(agent)
						self.propensities["behaviour_change"] += agent.behaviour_change_rate * self.behaviour_change_rate


		cpdef void recovery(self, double u, double t):
			cdef double tmp
			cdef int id_I 
			cdef Agent agent_I, agent
			cdef list tmp_list

			# select I or D to recover
			tmp_list = self.I_list + self.D_list
			tmp_sum = 0
			id_I = 0

			while tmp_sum < u:
				tmp_sum += tmp_list[id_I].recover_risk
				id_I += 1

			agent_I = tmp_list[id_I - 1]

			# check S round of agent
			# first round is incubation period
			# needs four rounds to recover
			if agent_I.I_round == 0:
				# incubation period ends
				# agent is now infectious 
				agent_I.become_infectious(t, self.recovery_rate)

				# increase infection risk for all S
				#self.r_inf = 0
				self.propensities["infection"] = 0
				for agent in self.S_list:
					agent.update_risk_added_I(agent_I)
					self.propensities["infection"] += agent.infection_risk * self.inf_rate

				# update diag propensity
				self.propensities["diagnosis"] += self.diag_rate

				# update recover propensity (only here)
				# remove incubation add recovery rate
				self.propensities["recovery"] += self.recovery_rate - self.incubation_rate

				# next round
				#agent_I.I_round += 1

			elif agent_I.I_round == self.number_infectious_compartments:
				# recover
				if agent_I.status == "I":
					# not infectious anymore
					# update propensities 
					self.propensities["infection"] = 0
					for agent in self.S_list:
						agent.update_risk_added_R(agent_I)
						self.propensities["infection"] += agent.infection_risk * self.inf_rate

					self.propensities["recovery"] -= self.recovery_rate
					self.propensities["diagnosis"] -= self.diag_rate

					agent_I.infectious = 0
					agent_I.recover_risk = 0

					self.I_list.pop(self.I_list.index(agent_I))

					if agent_I.behaviour_change_rate > 0 and self.propensities["behaviour_change"]>0:
						self.possible_measure_list.append(agent_I)
						self.propensities["behaviour_change"] += self.behaviour_change_rate * agent_I.behaviour_change_rate

				elif agent_I.status == "D":
					self.propensities["recovery"] -= self.recovery_rate
					agent_I.recover_risk = 0
					self.D_list.pop(self.D_list.index(agent_I))
					agent_I.lambda_plus = agent_I.parameters["lambda_plus"]



				# book keeping
				agent_I.status = "R"
				agent_I.recovery_time = t
				self.R_list.append(agent_I)
				self.total_immunized += 1

			else:
				# only move to next round of S
				agent_I.I_round += 1


		cpdef void measure(self, double u, double t):
			cdef int id_S 
			cdef Agent agent_S, agent
			cdef double b_c 


			if len(self.possible_measure_list) == 0:
				#self.r_measure = 0
				self.propensities["behaviour_change"] = 0
				return

			id_S = np.random.choice(len(self.possible_measure_list))

			agent_S = self.possible_measure_list[id_S]

			# change mu and T
			# take behaviour change from last simulation
			b_c = agent_S.b_c
			agent_S.behaviour_change_time = t


			if agent_S.status == "I":
				# change infection risk of all S
				self.propensities["infection"] = 0
				for agent in self.S_list:
					agent.update_risk_added_M(agent_S, b_c)
					self.propensities["infection"] += agent.infection_risk * self.inf_rate

				agent_S.lambda_plus *= b_c

			elif agent_S.status == "S":
				self.propensities["infection"] -= agent_S.infection_risk * self.inf_rate
				agent_S.lambda_plus *= b_c
				agent_S.get_infection_risk(self.I_list)
				self.propensities["infection"] += agent_S.infection_risk * self.inf_rate

			else:
				agent_S.lambda_plus *= b_c


			self.propensities["behaviour_change"] -= agent_S.behaviour_change_rate * self.behaviour_change_rate
			agent_S.behaviour_change_rate = 0
			self.active_measure_list.append(agent_S)
			self.possible_measure_list.pop(id_S)

			# update activation time for edge existence probability calculation
			#agent_S.behaviour_change_time = t

			# activate removal rate
			agent_S.behaviour_return = self.return_unvacc
			self.propensities["behaviour_return"] += self.return_unvacc


		cpdef void lift_measure(self, double u, double t):
			cdef int id_S 
			cdef Agent agent_S, agent


			if len(self.active_measure_list) == 0:
				self.propensities["behaviour_return"] = 0
				return

			id_S = np.random.choice(len(self.active_measure_list))
			agent_S = self.active_measure_list[id_S]


			# deactivate measure = return to unreduced contact behaviour
			if agent_S.status == "S":
				self.propensities["infection"] -= agent_S.infection_risk * self.inf_rate
				agent_S.lambda_plus = agent_S.parameters["lambda_plus"]
				agent_S.get_infection_risk(self.I_list)
				self.propensities["infection"] += agent_S.infection_risk * self.inf_rate
			elif agent_S.status == "I":
				#increase infection risk for S agents
				self.propensities["infection"] = 0
				for agent in self.S_list:
					# remove current risk
					agent.update_risk_removed_M(agent_S)
				agent_S.lambda_plus = agent_S.parameters["lambda_plus"]


			self.propensities["behaviour_return"] -= self.return_unvacc
			agent_S.behaviour_return = 0
			self.active_measure_list.pop(id_S)

			agent_S.behaviour_return_time = t

			# update activation time for edge existence probability calculation
			#agent_S.behaviour_change_time = t

