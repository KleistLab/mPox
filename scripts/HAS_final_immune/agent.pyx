cdef class Agent:
	def __init__(self, int i, int mu, double mu_1, double lambda_plus, double lambda_minus, int behaviour_somewhat, int behaviour_strong, double inf_rate, double b_c):
		self.id = i
		self.status = "S"
		self.parameters = {'lambda_plus': lambda_plus, 'lambda_minus': lambda_minus, 'mu_i': mu,
						  'behaviour_strong': behaviour_strong, 'behaviour_somewhat': behaviour_somewhat}
		self.lambda_plus = lambda_plus
		self.lambda_minus = lambda_minus
		self.inf_rate = inf_rate
		self.infection_risk = 0
		self.diagnosis_risk = 0
		self.recover_risk = 0
		self.infection_time = 0
		self.infectious = 0
		self.behaviour_change_rate = behaviour_strong + behaviour_somewhat/2
		self.behaviour_return = 0
		self.I_round = 0
		self.mu_1 = mu_1
		self.behaviour_change_time = 0
		self.behaviour_return_time = 0
		self.b_c = b_c
		self.exposure_time = 0
		self.diagnosis_time = 0
		self.recovery_time = 0
		#self.sus_risk = 0

	cdef public:
		cpdef void get_infection_risk(self, list I_list):
			cdef Agent agent
			cdef double res

			res = 0

			for agent in I_list:
				res += agent.infectious * self.lambda_plus*agent.lambda_plus/(self.lambda_plus*agent.lambda_plus + self.lambda_minus*agent.lambda_minus)

			self.infection_risk = self.inf_rate * res

		cpdef Agent select_infected_partner(self, list I_list, double u):
			cdef Agent agent
			cdef double res
			cdef int id_I = 0

			res = 0
			u *= 1/self.inf_rate
			agent = I_list[0]

			while res < u:
				agent = I_list[id_I]
				res += agent.infectious * self.lambda_plus*agent.lambda_plus/(self.lambda_plus*agent.lambda_plus + self.lambda_minus*agent.lambda_minus)
				id_I += 1

			return(agent)

		cpdef void update_risk_added_I(self, Agent agent):
			self.infection_risk += agent.infectious * self.inf_rate * self.lambda_plus*agent.lambda_plus/(self.lambda_plus*agent.lambda_plus + self.lambda_minus*agent.lambda_minus)

		cpdef void update_risk_added_R(self, Agent agent):
			self.infection_risk -= agent.infectious * self.inf_rate * self.lambda_plus*agent.lambda_plus/(self.lambda_plus*agent.lambda_plus + self.lambda_minus*agent.lambda_minus)


		cpdef void update_risk_added_M(self, Agent agent, double beta):
			self.infection_risk -= agent.infectious * self.inf_rate * self.lambda_plus* agent.lambda_plus/(self.lambda_plus* agent.lambda_plus + self.lambda_minus*agent.lambda_minus)
			self.infection_risk += agent.infectious * self.inf_rate * self.lambda_plus* beta * agent.lambda_plus/(self.lambda_plus* beta * agent.lambda_plus + self.lambda_minus*agent.lambda_minus)

		cpdef void update_risk_removed_M(self, Agent agent):
			self.infection_risk -= agent.infectious * self.inf_rate * self.lambda_plus* agent.lambda_plus/(self.lambda_plus* agent.lambda_plus + self.lambda_minus*agent.lambda_minus)
			self.infection_risk += agent.infectious * self.inf_rate * self.lambda_plus* agent.parameters["lambda_plus"]/(self.lambda_plus* agent.parameters["lambda_plus"] + self.lambda_minus*agent.lambda_minus)

		cpdef void become_infected(self, double incubation_rate):
			self.status = "I"
			self.inf_rate = 0
			self.infection_risk = 0
			self.recover_risk = incubation_rate

		cpdef void become_infectious(self, double t, double recover_rate):
			self.infection_time = t
			self.infectious = 1
			self.recover_risk = recover_rate
			self.I_round = 1


		cpdef void become_diagnosed(self):
			self.status = "D"
			self.diagnosis_risk = 0
			self.lambda_plus = 0
			self.behaviour_change_rate = 0

		cpdef void become_recovered(self):
			self.status = "R"
			#self.lambda_plus = self.parameters["lambda_plus"]
			self.infection_risk = 0
			self.diagnosis_risk = 0
			self.infection_time = 0
			self.infectious = 0
			#self.sus_risk = self.parameters["sus_i"]

		