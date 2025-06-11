import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy
import sys
import os
import math
import pandas as pd
from tqdm import tqdm

import time


class pathway_agents_v11():

    def __init__(self, param):

        # saving all the parameters to feed them to sim_data
        self.param = param

        # agent variables
        self.rwd_prob = 1
        self.pun_prob = 1
        self.rwd_mag = param['rwd_mag']
        self.pun_mag = param['pun_mag']
        
        # number of episodes
        self.n_episodes = param['n_episodes']
        self.n_test_eps = param['n_test_eps']
        self.n_test_episodes = 0
        
        # reinforcement learning parameters
        self.beta = param['beta']
        self.gamma_v = param['gamma_v']
        self.gamma_dm = param['gamma_dm']
        self.n_states = param['n_states']
        self.second_tone_list = param['second_tone_list']
        self.alpha_i = param['alpha_i']
        self.n_actions = 3
        self.n_pathways = 2

        self.env_param = param['env_param']
        self.t_dwell_ctr = self.env_param[0]
        self.t_dwell_var = self.env_param[1]
        self.t_dwell_min = self.env_param[2]
        self.t_dwell_max = self.env_param[3]
        self.n_trajs = self.env_param[4]
        self.t_dwell_list = self.env_param[5]
        
        # pathway weights - for learning
        self.w_D = param['omega'][0]
        self.w_I = param['omega'][1]
        self.w_CD = param['omega'][2]
        self.w_CI = param['omega'][3]
        
        # pathway weights - for testing
        self.tw_D = param['omega'][0]
        self.tw_I = param['omega'][1]
        self.tw_CD = param['omega'][2]
        self.tw_CI = param['omega'][3]
        

        # observer variables // breaking fixation parameters
        self.obs_var1 = param['obs_var'][0] # obs1 noise
        self.obs_var2 = param['obs_var'][1] # obs2 noise
        
        # transfer function parameters
        dp_dls_param = param['transfer_param'][0]
        ip_dls_param = param['transfer_param'][1]
        dp_dms_param = param['transfer_param'][2]
        ip_dms_param = param['transfer_param'][3]

        
        # number of observations we're blocking from the updates
        self.n_blocked_states = 0
        self.episode_training_bias = 0.7
        self.bias_episode_ratio = 0.5
        
        # normalised state visits
        self.n_sv = 0.1

        # Optogenetic perturbation variables (need to change episode and t to self.curr_ep and self.curr_st)\
        self.curr_st = 0 # current state
        self.perturbation_type = False
        self.pert_mag = 0
        
        # learning rates for actor and critic - smaler alphas -> more iterations to converge
        self.alpha_v_cap = 0.01
        self.alpha_a_cap = 0.001 # trying to use same alphas for both actor and critic
        self.alpha_a3 = param['alpha3']
        #self.alpha_a_cap = 0.01 # trying to use same alphas for both actor and critic

        # importing the environment
        self.get_environment()
        self.set_nl_td_parameters(dp_dls_param, ip_dls_param, dp_dms_param, ip_dms_param)
        self.use_solution_flag = False

        # test agent
        self.trial_subset = []
        self.n_test_episodes = 0

    """
    ENVIRONMENT IMPORT
    This function imports the environment/task to be used later in training
    """
    def get_environment(self):

        # this function calls the timing task classical conditioning environment only

        from environments.timing_task_csc import timing_task_csc_v8
        
        # list of environment variables
        # n_states, second_tone_list, rwd_mag, pun_mag, rwd_prob, dwell_time_list
        self.env = timing_task_csc_v8(self.n_states, self.second_tone_list,
                                      self.rwd_mag, self.rwd_prob, self.pun_mag, self.pun_prob, self.t_dwell_list)

    """
    TRANSFER FUNCTIONS
    These functions represent the transfer functions of the Reward Prediction Error used 
    """

    # non-linear td transfer functions 

    # DLS dMSN transfer function
    def nl_tdp(self, x):
        return self.e_tdp + self.a_tdp / (self.b_tdp + self.c_tdp * np.exp(-self.d_tdp * x + self.d_tdp))
    
    # DLS iMSN transfer function
    def nl_tdn(self, x): # delay period transfer function
        return self.e_tdn + self.a_tdn / (self.b_tdn + self.c_tdn * np.exp(self.d_tdn * x + self.d_tdn))

    # DMS dMSN transfer function
    def nl_tdp2(self, x):
        return self.e_tdp2 + self.a_tdp2 / (self.b_tdp2 + self.c_tdp2 * np.exp(-self.d_tdp2 * x + self.d_tdp2))

    # DMS iMSN transfer function
    def nl_tdn2(self, x):
        return self.e_tdn2 + self.a_tdn2 / (self.b_tdn2 + self.c_tdn2 * np.exp(self.d_tdn2 * x + self.d_tdn2))
    
    
    # CORRECT THIS AND REMOVE SPURIOUS PARAMETERS
    def set_nl_td_parameters(self, p_tdp, p_tdn, p_tdp2, p_tdn2):
        # setting the parameters for the transfer function
        # make this vectorial for godssake

        # DLS dMSN
        self.a_tdp = p_tdp[0]
        self.b_tdp = p_tdp[1]
        self.c_tdp = p_tdp[2]
        self.d_tdp = p_tdp[3]
        self.e_tdp = p_tdp[4]

        # DLS iMSN
        self.a_tdn = p_tdn[0]
        self.b_tdn = p_tdn[1]
        self.c_tdn = p_tdn[2]
        self.d_tdn = p_tdn[3]
        self.e_tdn = p_tdn[4]

        # DMS dMSN
        self.a_tdp2 = p_tdp2[0]
        self.b_tdp2 = p_tdp2[1]
        self.c_tdp2 = p_tdp2[2]
        self.d_tdp2 = p_tdp2[3]
        self.e_tdp2 = p_tdp2[4]

        # DMS dMSN
        self.a_tdn2 = p_tdn2[0]
        self.b_tdn2 = p_tdn2[1]
        self.c_tdn2 = p_tdn2[2]
        self.d_tdn2 = p_tdn2[3]
        self.e_tdn2 = p_tdn2[4]
        
    
    # this doesn't necessarily need to be here
    def plot_transfer_function(self, size):
        x = np.linspace(-10, 10, 100)

        columns = 1
        rows = 1
        fig = plt.figure(figsize=(size[0], size[1]))

        gs = gridspec.GridSpec(columns, rows)
        gs.update(hspace=0.5)

        ax = fig.add_subplot(gs[0, 0])

        #fig,ax = plt.figure(figsize = (size[0], size[1]))
        ax.plot(x, self.nl_tdp(x), 'b', label='dMSN-DLS')
        ax.plot(x, self.nl_tdn(x), 'r', label='iMSN-DLS')
        ax.plot(x, self.nl_tdp2(x), 'b', label='dDMSN-DMS', linestyle = 'dotted')
        ax.plot(x, self.nl_tdn2(x), 'r', label='iMSN-DMS',linestyle = 'dotted')

        ax.axvline(x=0, color='k', linewidth='0.7')
        ax.axhline(y=0, color='k', linewidth='0.7')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set(xlabel='$\delta_t$', ylabel='$f ^ p(\delta_t)$',title='RPE transfer function')
        ax.legend()
        
        return fig, self.nl_tdp(x), self.nl_tdp2(x), self.nl_tdn(x), self.nl_tdn2(x)

    """
    USEFUL FUNCTIONS
    """
    # this function doesn't need to be here, it's only called in the train_agent
    def min_max_norm(self, vec):
        return (vec-np.min(vec))/(np.max(vec)-np.min(vec))
    """
    POLICIES
    """

    def p_policy(self, A, beta):
        
        # receives A = w_D A_D + w_I A_I  + w_c A_C- full action-value function
        # dim: 2,3 - pathways, s_t, actions
        # only uses indexes (0,1) - (L,R) actions
        # A = A[0:2]
        
        x = A - A.max(axis=None, keepdims=True)
        y = np.exp(x * beta)

        
        # condition for hold action to be selected depends only on the values of L,R actions
        # hold action is selected if Act < 0 - implies -> indirect pathway > direct + dorso-medial
        if A[0] < 0 and A[1] < 0: 
            y[2] = 1
        else:
            y[2] = 0

        # replacing infinite values by 1
        for i, e in enumerate(y):
            if np.isinf(e):
                y[i] = 1

        # normalizing to get probabilities
        action_prob = y / y.sum(axis=None, keepdims=True)
        selected_action = np.argmax(np.random.multinomial(1, action_prob, size=1))

        return selected_action

    """
    OBSERVER DYNAMICS
    """
    # used in both train and test agent functions
    def update_obs_log(self, obs_log, new_state):
        obs_log = np.roll(obs_log,-1)
        obs_log[-1] = new_state
        return obs_log.astype(int)
    
    
    def onehot(self, value, max_value):
        
        vec = np.zeros(max_value)
        vec[value] = 1
        
        return vec
    
    """
    TRAINING ALGORITHMS
    """
    
    def train_agent(self):

        # States and Observations
        current_state = 0
        next_state = 0
        
        previous_obs = 0
        current_obs = 0
        next_obs = 0
        
        current_obs2 =0
        next_obs2 = 0

        reward = 0
        action = 0
        previous_action = 0

        td_list = []
        idx_list = []

        # Loading the environment
        env = self.env

        # Convergence measures
        pV = np.zeros(env.n_total_states)
        pVc = np.zeros(env.n_total_states)
        pA = np.zeros((self.n_pathways, env.n_total_states, self.n_actions))
        diffV = np.zeros(self.n_episodes)
        diffVc = np.zeros(self.n_episodes)
        diffA = np.zeros((self.n_episodes, self.n_pathways, self.n_actions))
        diffAc = np.zeros((self.n_episodes, self.n_pathways, self.n_actions))
        
        # value functions for the simulations
        V = np.zeros(env.n_total_states)  # state value function
        Vc = np.zeros(env.n_total_states)  # state value function
        A = np.random.uniform(0, 0.01, (self.n_pathways, env.n_total_states, self.n_actions))  # Direct indirect pathway value functions     
        Ac = np.random.uniform(0, 0.01, (self.n_pathways, env.n_total_states, self.n_actions)) # 3rd pathway, with 2 pathways as well
        Act = np.zeros((env.n_total_states, self.n_actions)) # full advantage

        # behavioural measures
        choices = np.zeros(self.n_test_eps)
        # saving behaviour in true time
        true_time = np.arange(0.6,2.4,0.01) # parameterise this
        behaviour_tt = (-1) * np.ones((self.n_episodes, true_time.shape[0])) # size of the true time binning

        behaviour_h = (-1) * np.ones((self.n_episodes, env.n_total_states)) #episode_number, trial_id, action // initialised at -1
        choice_v = np.zeros((self.n_episodes, env.n_total_states)) #episode_number, trial_id - (0,1,2,3) - (hodl, premature, correct, incorrect)
        trial_h = np.zeros((self.n_episodes,2))
        rwd_hist = np.zeros((self.n_episodes, env.n_total_states))
        state_visits = [np.zeros((2,env.n_total_states)), np.zeros((env.n_total_states,self.n_actions))]
        
        # critic and actor parameters
        delta = 0 # global reward prediction error
        delta_c = 0 # global reward prediction error
        delta_h = np.zeros((4, self.n_episodes, env.n_total_states))

        self.alpha_v = np.ones(env.n_total_states)
        self.alpha_a = np.ones(env.n_total_states)
        self.alpha_a2 = np.ones(env.n_total_states)
        
        self.gamma_td = 0.98
        self.gamma_td_list = [1,0.98, 0.97, 0.96, 0.95, 0.94]
        self.alpha_td = 0.01
        
       # state transition matrices
        M0 = np.zeros((self.env.n_total_states, self.env.n_total_states, 3))
        M1 = np.zeros((3, self.env.n_total_states, self.env.n_total_states))
        M_list = np.zeros((len(self.gamma_td_list), 3, self.env.n_total_states, self.env.n_total_states))
        
        state_data = []

        # second tone list array
        stl = np.asarray(env.second_tone_list)

        print('running episodes')
        print('training bias is ', str(self.training_episode_bias), ' for the first ',  self.n_episodes * self.bias_episode_ratio)

        for episode in tqdm(range(self.n_episodes), desc="Training Episodes", unit="episode"):


            current_tone = np.random.choice(np.arange(len(env.second_tone_list)))

            tr_i = np.random.normal(self.t_dwell_ctr, self.t_dwell_var) # sample a dwell time from a gaussian
            dwell_t, trial_n = env.find_nearest(self.t_dwell_list, tr_i) # finding the closest dwell time to the one sampled

            # resampling in case we hit the edges
            if trial_n == 0 or trial_n == len(self.t_dwell_list):
                tr_i = np.random.normal(self.t_dwell_ctr, self.t_dwell_var) 
                dwell_t, trial_n = env.find_nearest(self.t_dwell_list, tr_i)
                current_trial = trial_n
            else:
                current_trial = trial_n

            td_list.append(self.t_dwell_list[current_trial]) # saving the dwell times 
            idx_list.append(current_trial) # saving the indices from which they sprout
            
            # previous dwell time selection
            #current_trial = np.random.choice(np.arange(env.n_trials_per_tone))

            trial_id = [current_tone, current_trial]
            trial_h[episode][0] = current_tone
            trial_h[episode][1] = current_trial
        
            # value functions comparison over episodes
            pV = copy.copy(V)
            pVc = copy.copy(Vc)
            pA = copy.copy(A)
            pAc = copy.copy(Ac)

            # state and observation initialisations
            current_state = 0
            current_state2 = 0
            current_obs = 0
            current_obs2 = 0
            
            # optogenetic perturbation variables

            for t in range(env.n_states):
                
                # FULL ADVANTAGE FUNCTION ------------------------------------------------------
                
                Act[current_obs, :] = self.w_D * A[0, current_obs, :] - self.w_I * A[1, current_obs, :] + self.w_CD * Ac[0, current_obs2, :] - self.w_CI * Ac[1, current_obs2, :]
                
                # POLICY -------------------------------------------------------------------
                
                if self.use_solution_flag == False:
                    action = self.p_policy(Act[current_obs, :], self.beta)
                else:
                    print(current_obs, current_tone, current_trial, env.trial_st[current_tone, current_trial], np.where(env.trial_st[current_tone, current_trial] == current_obs)[0])
                    opt_act_st = np.where(env.trial_st[current_tone, current_trial] == current_obs)[0][0]
                    action = env.opt_act[current_tone, current_trial][opt_act_st]

                # ENVIRONMENT STATE UPDATE --------------------------------------------------
                
                next_state, next_state2, reward, choice_type, terminal_flag = env.get_outcome(trial_id, current_state, current_state2, action)
                
                # BEHAVIOUR HISTORY----------------------------------------------------------
                
                behaviour_h[episode, current_state] = action
                rwd_hist[episode, current_state] = reward
                choice_v[episode, current_state] = choice_type
                
                ### OBSERVATION MODEL --------------------------------------------------------
                # without blocking the middle state transitions
                # turn this into a function as well
                
                eta1 = int(np.random.normal(0, self.obs_var1))
                #eta2 = int(np.random.normal(0, self.obs_var2))
                eta2 = eta1 # if noise is selected then both agents share it
                # remember the uncorrelated noise
                    
                # avoiding the intermediate observations
                # initial state boundary
                if next_state + eta1 < 0:
                    next_obs = 0
                    
                # 2nd tone boundaries
                
                # pre second tone
                elif next_state < self.n_states:
                    if next_state + eta1 > self.n_states - self.n_blocked_states:
                        next_obs = self.n_states - self.n_blocked_states # go back to pre-second tone states
                    else:
                        next_obs = next_state + eta1
                        
                # post second tone
                elif next_state >= self.n_states:
                    if next_state + eta1 < env.n_total_states - 1:
                        if next_state + eta1 < self.n_states + self.n_blocked_states:
                            next_obs = self.n_states + self.n_blocked_states
                        else:
                            next_obs = next_state + eta1
                    else:
                        next_obs = env.n_total_states - 1
                
                
                # DORSO-MEDIAL OBSERVER
                if next_state2 + eta2 < 0:
                    next_obs2 = 0
                
                # pre second tone
                elif next_state2 < self.n_states:
                    if next_state2 + eta2 > self.n_states - self.n_blocked_states:
                        next_obs2 = self.n_states - self.n_blocked_states # go back to pre-second tone states
                    else:
                        next_obs2 = next_state2 + eta2

                ### OBSERVATION VECTORS ------------------------------------------------------------
                
                obs_vec = [current_obs, next_obs, current_obs2, next_obs2, action, previous_action]
                
                # updating the state transition matrix 

                M0[current_obs, next_obs, action] += 1

                # update the real successor representation
                next_action = self.p_policy(Act[next_obs, :], self.beta)
                I = self.onehot(current_obs, self.env.n_total_states)
                td_error = (I + self.gamma_td * M1[next_action, next_obs, :] - M1[action, current_obs, :])
                M1[action, current_obs, :] += self.alpha_td * td_error
                
                for i,gamma in enumerate(self.gamma_td_list):
                    td_error = (I + gamma * M1[next_action, next_obs, :] - M1[action, current_obs, :])
                    M_list[i, action, current_obs, :] += self.alpha_td * td_error
                
                
                ### VALUE UPDATE ----------------------------------------------------------------
                
                state_data.append([obs_vec, eta1, eta2])
                
                delta = reward + self.gamma_v * V[next_obs] - V[current_obs]           
                delta_c = reward + self.gamma_dm * Vc[next_obs2] - Vc[current_obs2]           

                V, Vc, A, Ac, state_visits = self.update_value_functions(obs_vec, reward, delta, delta_c, V, Vc, A, Act, Ac, state_visits, trial_id)
                
                ### CONVERGENCE MEASURES ----------------------------------------------------------------
                
                if episode > self.n_episodes - self.n_test_eps:
                    if reward < 0:
                        choices[self.n_episodes - episode - self.n_test_eps] = 0
                    if reward > 0:
                        choices[self.n_episodes - episode - self.n_test_eps] = 1 
                    
                ### UPDATE STATES ----------------------------------------------------------------
                current_state = next_state
                current_state2 = next_state2

                current_obs = next_obs

                current_obs2 = next_obs2
                previous_action = action

                if terminal_flag:        
                #if current_state == env.n_total_states - 1:        
                    
                    # value function change per episode
                    diffV[episode] = np.sum((V-pV)**2)
                    diffVc[episode] = np.sum((Vc-pVc)**2)
                    for p in range(self.n_pathways):
                        for a in range(self.n_actions):
                            diffA[episode, p, a] = np.sum((A[p, :, a] - pA[p, :, a])**2)
                            diffAc[episode,p, a] = np.sum((Ac[p,:, a] - pAc[p,:, a])**2)                    
                    break

                # end trial if there was a wrong action
                
                    
        
        # data to plot the transfer function
        x_tf = np.linspace(-10, 10, 100)
        y_tdp = self.nl_tdp(x_tf)
        y_tdp2 = self.nl_tdp2(x_tf)
        y_tdn = self.nl_tdn(x_tf)
        tf_vec = [x_tf, y_tdp, y_tdp2, y_tdn]
                
        data_dict = {'V': V, 'Vc': Vc, 'state_visits': state_visits[0][0], 'A': A, 'Act': Act, 'Ac' : Ac,
                    'diffV': diffV, 'diffA': diffA, 'diffAc': diffAc, 'choices': choices, 
                    'trial_h' : trial_h, 'behaviour_h' : behaviour_h, 'rwd_hist':rwd_hist, 
                    'state_visits_full' : state_visits, 'state_data': state_data,
                    'delta_h' : delta_h, 'trial_h':trial_h, 'param': self.param, 
                    'tf_vec' : tf_vec, 'z_vec':env.z_vec, 'choice_v' : choice_v, 'idx_list' : idx_list,
                     'td_list' : td_list, 'M0': M0, 'M1' : M1, 'M_list': M_list, 'gamma_td_list' : self.gamma_td_list}

        return data_dict

    
    # 2 PATHWAY DMS ACTION PREFERENCES
    
    def update_value_functions(self, obs_vec, reward, delta, delta_c, V, Vc, A, Act, Ac, state_visits, trial_id):

        # this function is performing a lot of calculations that might not be necessary to perform
        # try to clean and remove unecessary lines
        
        # unpacking obsevation vector
        current_obs = obs_vec[0] 
        next_obs = obs_vec[1]
        current_obs2 = obs_vec[2]
        next_obs2 = obs_vec[3] # unnecessary
        action = obs_vec[4]
        previous_action = obs_vec[5]
        
        # LEARNING RATES -------------------------------------------------------------
        state_visits[0][0][current_obs] += 1
        state_visits[0][1][current_obs2] += 1
        state_visits[1][current_obs2, action] += 1

        inv_sv = self.alpha_i / state_visits[0][0][current_obs]
        inv_sv2 = self.alpha_i / state_visits[0][1][current_obs2]
        
        if inv_sv < self.alpha_v_cap:
            self.alpha_v[current_obs] = inv_sv
        else:
            self.alpha_v[current_obs] = self.alpha_v_cap
        
        if inv_sv2 < self.alpha_a_cap:
            self.alpha_a2[current_obs2] = inv_sv2
        else:
            self.alpha_a2[current_obs2] = self.alpha_a_cap
        
        # Critic, direct and indirect pathways
        if current_obs < self.env.n_total_states - 1:
            # calculating this as a bootstrap but not using it for the Q values
            V[current_obs] += self.alpha_v[current_obs] * delta
            Vc[current_obs2] += self.alpha_v[current_obs2] * delta_c

            if action != 2:
                A[0, current_obs, action] += self.alpha_v[current_obs] * self.nl_tdp(delta)
                A[1, current_obs, action] += self.alpha_v[current_obs] * self.nl_tdn(delta)
                
        elif current_obs == self.env.n_total_states - 1:
            V[current_obs] = 0
            A[0, current_obs, action] = 0
            A[1, current_obs, action] = 0
        
        # dorso-medial pathway update functions
        if current_obs2 < self.env.n_total_states - 1:
            if action != 2:
                Ac[0, current_obs2, action] += self.alpha_v[current_obs2] * self.nl_tdp(delta_c)
                Ac[1, current_obs2, action] += self.alpha_v[current_obs2] * self.nl_tdn(delta_c)
            
        elif current_obs2 == self.env.n_total_states - 1:
            Ac[0, current_obs2, action] = 0
            Ac[1, current_obs2, action] = 0

        return V, Vc, A, Ac, state_visits


    def test_agent(self, sim_data):

        #self.n_test_episodes = 0

        # States and Observations
        current_state = 0
        next_state = 0
        
        previous_obs = 0
        current_obs = 0
        next_obs = 0
        
        current_obs2 =0
        next_obs2 = 0

        reward = 0
        action = 0
        previous_action = 0

        # Loading the environment
        env = self.env
        
        # loading previously learned value functions
        V = sim_data['V']
        Vc = sim_data['Vc']
        A = sim_data['A']
        Ac = sim_data['Ac']
        #Act = sim_data['Act'] # not taking the simulated one in order to amplify BF
        Act = np.zeros((env.n_total_states, self.n_actions)) # full advantage
        
        # successor matrices
        
        M0 = np.zeros((self.env.n_total_states, self.env.n_total_states, 3))
        M1 = np.zeros((3, self.env.n_total_states, self.env.n_total_states))
            
        # behavioural measures
        choices = np.zeros(self.n_test_eps) # ???
        behaviour_h = (-1) * np.ones((self.n_test_episodes, env.n_total_states)) #episode_number, trial_id, action // initialised at -1
        choice_v = np.zeros((self.n_test_episodes, env.n_total_states)) #episode_number, trial_id - (0,1,2,3) - (hodl, premature, correct, incorrect)
        trial_h = np.zeros((self.n_test_episodes,2))
        rwd_hist = np.zeros((self.n_test_episodes, env.n_total_states))
        state_visits = [np.zeros((2,env.n_total_states)), np.zeros((env.n_total_states,self.n_actions))]
        
        state_data = []

        if self.perturbation_type:
            print(self.perturbation_type)

        print('running episodes')

        for episode in range(self.n_test_episodes):

            if episode == int(self.n_test_episodes/3):
                print('30%')

            if episode == int(self.n_test_episodes/2):
                print('50%')
            
            if episode == int(3 * self.n_test_episodes/4):
                print('75%')


            # saving trial history
            # randomly sampled trials no biasing
            current_tone = np.random.choice(self.trial_subset)

            # which trial is the center trial
            # define a gaussian centered on that and find the closest integer
            
            current_trial = np.random.choice(np.arange(env.n_trials_per_tone))
            trial_id = [current_tone, current_trial]
            trial_h[episode][0] = current_tone
            trial_h[episode][1] = current_trial

            # state and observation initialisations
            current_state = 0
            current_state2 = 0
            current_obs = 0
            current_obs2 = 0

            for t in range(env.n_states):

                # FULL ADVANTAGE FUNCTION ------------------------------------------------------
                # Perturbations
                xA = copy.copy(A)
                xAc = copy.copy(Ac)
                
                # change perturbation nomenclature
                # iDLS-BL
                # iDLS-CS
                # iDLS-CL
                # dDMS-BL
                # dDMS-CS
                # dDMS-CL
                
                if self.perturbation_type == 'iDLS-BL':                        
                    xA[1, current_obs, :] = self.pert_mag * xA[1, current_obs, :]

                if self.perturbation_type == 'iDLS-CS':                        
                    xA[1, current_obs, 0] = self.pert_mag * xA[1, current_obs, 0]

                if self.perturbation_type == 'iDLS-CL':                        
                    xA[1, current_obs, 1] = self.pert_mag * xA[1, current_obs, 1]

                if self.perturbation_type == 'dDMS-BL':                        
                    xAc[0, current_obs2, :] = self.pert_mag * xAc[0, current_obs2, :]

                if self.perturbation_type == 'dDMS-CS':                        
                    xAc[0, current_obs2, 0] = self.pert_mag * xAc[0, current_obs2, 0]

                if self.perturbation_type == 'dDMS-CL':                        
                    xAc[0, current_obs2, 1] = self.pert_mag * xAc[0, current_obs2, 1]
                    
                Act[current_obs, :] = self.tw_D * xA[0, current_obs, :] - self.tw_I * xA[1, current_obs, :] + self.tw_CD * xAc[0, current_obs2, :] - self.tw_CI * xAc[1, current_obs2, :]
                
                # POLICY -------------------------------------------------------------------
                
                action = self.p_policy(Act[current_obs, :], self.beta)

                # ENVIRONMENT STATE UPDATE --------------------------------------------------
                
                next_state, next_state2, reward, choice_type, terminal_flag = env.get_outcome(trial_id, current_state, current_state2, action)
                
                # BEHAVIOUR HISTORY----------------------------------------------------------
                
                behaviour_h[episode, current_state] = action
                rwd_hist[episode, current_state] = reward
                choice_v[episode, current_state] = choice_type
                
                
                ### OBSERVATION MODEL --------------------------------------------------------
                ## THIS SECTION NEEDS TO BE UPDATED - ETA IS UNNECESSARY - DEFINED AS ZERO IN SIMULATIONS
                
                eta1 = int(np.random.normal(0, self.obs_var1))
                eta2 = eta1

                # zero boundary
                if next_state + eta1 < 0:
                    next_obs = 0
                    
                # 2nd tone boundaries
                
                # pre second tone
                elif next_state < self.n_states:
                    if next_state + eta1 > self.n_states - self.n_blocked_states:
                        next_obs = self.n_states - self.n_blocked_states # go back to pre-second tone states
                    else:
                        next_obs = next_state + eta1
                        
                # post second tone
                elif next_state >= self.n_states:
                    if next_state + eta1 < env.n_total_states - 1:
                        if next_state + eta1 < self.n_states + self.n_blocked_states:
                            next_obs = self.n_states + self.n_blocked_states
                        else:
                            next_obs = next_state + eta1
                    else:
                        next_obs = env.n_total_states - 1
                
                # DORSO-MEDIAL OBSERVER
                if next_state2 + eta2 < 0:
                    next_obs2 = 0
                
                # pre second tone
                elif next_state2 < self.n_states:
                    if next_state2 + eta2 > self.n_states - self.n_blocked_states:
                        next_obs2 = self.n_states - self.n_blocked_states # go back to pre-second tone states
                    else:
                        next_obs2 = next_state2 + eta2
                    
                ### STATE VISIT UPDATE

                state_visits[0][0][current_obs] += 1
                state_visits[0][1][current_obs2] += 1
                state_visits[1][current_obs2, action] += 1
                
                ### SUCCESSOR MATRICES
                
                M0[current_obs, next_obs, action] += 1

                # update the real successor representation
                next_action = self.p_policy(Act[next_obs, :], self.beta)
                I = self.onehot(current_obs, self.env.n_total_states)
                td_error = (I + self.gamma_td * M1[next_action, next_obs, :] - M1[action, current_obs, :])
                M1[action, current_obs, :] += self.alpha_td * td_error

                ### UPDATE STATES ----------------------------------------------------------------
                current_state = next_state
                current_state2 = next_state2
                current_obs = next_obs
                current_obs2 = next_obs2
                previous_action = action

                # append to the dataframe after the trial is over
                if terminal_flag:
                    terminal_flag = False
                    break
                
        data_dict = {'n_episodes' : self.n_test_episodes, 'choices': choices, 'trial_h' : trial_h, 
                    'behaviour_h' : behaviour_h, 'rwd_hist':rwd_hist, 
                    'state_visits_full' : state_visits, 'state_data': state_data,
                    'trial_h':trial_h, 'pert_type' :self.perturbation_type,
                    'pert_mag' : self.pert_mag, 'param' : self.param, 'choice_v' : choice_v,
                    'M0': M0, 'M1' : M1}

        return data_dict
