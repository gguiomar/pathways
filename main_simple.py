
#%%

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy
import scipy as sp
from scipy import ndimage
from scipy.optimize import curve_fit
import sys
import os
from os import walk
import pandas as pd
import pickle
import datetime
from IPython.display import clear_output, display
from numpy import linalg as LA
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from matplotlib import rc

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'axes.linewidth': 2}
plt.rcParams.update(params)
plt.rcParams.update({'font.size': 14})
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 2

#pathways agent and analysis imports
from agents.pathway_agents import pathway_agents_v11
from analysis.pathway_analysis import pathway_analysis_v6

# importing simulation and analysis code
cwd = os.getcwd()
fig_path = cwd
data_path = os.path.join(cwd, 'simulation_data')

#%% TRAINING 
#n_eps = 250000
n_eps = 10000
n_test_eps = int(n_eps * 0.7)

# 3rd observer variables 
obs_var = [0,0] # horizontal noise for the 2 observers - not using right now
omega_c = 1
alpha3 = 0.01 # trying out a scaling learning rate

# there's an extra state added here due to a bug in the psychometric
stl = [0.1, 0.3, 0.5, 0.6000, 0.8, 1.0500, 1.2600, 1.3800, 1.6200, 1.7400, 1.9500, 2.2, 2.4000, 2.5000]

t_dwell_var = 23 
t_dwell_min = 65 
t_dwell_ctr = 70 
t_dwell_max = 100 

t_dwell_list = np.arange(t_dwell_min, t_dwell_max, 3)
n_trajs = t_dwell_list.shape[0]
print('Number of trajectories', n_trajs)
env_param = [t_dwell_ctr, t_dwell_var, t_dwell_min, t_dwell_max, n_trajs, t_dwell_list]

# dls parameters
dp_dls_param = [11.5, 1, 0.9, 1, -3.5, 0]
ip_dls_param = [11.5, 1, 0.9, 1, -3.5, 0]

# dms parameters
dp_dms_param = [11.5, 1, 0.9, 1, -3.5, 0]
ip_dms_param = [11.5, 1, 0.9, 1, -3.5, 0]

train_sim_notes = "Tranining with split DMS and symmetric transfer functions"

param = {'n_states' : 40, 'beta' : 1.5, 'gamma_v' : 0.98, 'gamma_dm' : 0.98, 'rwd_mag' : 10, 'pun_mag' : -5, 
           'n_episodes' : n_eps, 'n_test_eps' : n_test_eps, 
           'second_tone_list' : stl, 'alpha_i' : 20, 'env_param' : env_param, 'obs_var' : obs_var, 
           'omega' : [1,1, 0.5,0.5], 'transfer_param' : [dp_dls_param, ip_dls_param, dp_dms_param, ip_dms_param],
           'alpha3' : alpha3, 'sim_notes' : train_sim_notes}

fsize = [5,5]
tf_m = pathway_agents_v11(param)

tf_m.set_nl_td_parameters(dp_dls_param, ip_dls_param, dp_dms_param, ip_dms_param)
tf_m.env.jump_correction = 0
tf_m.env.generate_environment()
tf_m.n_blocked_states = 5
tf_m.training_episode_bias = 0.5
tf_m.bias_episode_ratio  = 0.5
tf_m.env.plot_env_trajectories([5,5], tf_m.env.z_vec)


# dls parameters
dp_dls_param = [11.5, 1, 2, 1, -2, 0]
ip_dls_param = [11.5, 1, 2, 1, -2, 0]

# dms parameters
dp_dms_param = [11.5, 1, 2, 1, -2, 0]
ip_dms_param = [11.5, 1, 2, 1, -2, 0]

tf_m.set_nl_td_parameters(dp_dls_param, ip_dls_param, dp_dms_param, ip_dms_param)
tf_fig2 = tf_m.plot_transfer_function([5,3])


#%%
sim_data = tf_m.train_agent()

#%%

f_size = [12,10]
init_f = 0.7
tf_a = pathway_analysis_v6(sim_data, param)
n_bins = 15
f1 = tf_a.grid_plot_training_2C4A(f_size, '2C3A', sim_data, param, tf_m, init_f, n_bins)

#%%

n_timepoints = 13
fig_size = [8,7]
f2 = tf_a.grid_plot_neural_activity_DMS_DLS(f_size, sim_data, n_timepoints, tf_m)

# %%


# TEST

tf_m.n_test_episodes = 10000
tf_m.perturbation_type = False
tf_m.trial_subset = [3,5,6,7,8,9,10,11,12] # this should be tone subset not trial subset
tone_list = [param['second_tone_list'][i] for i in tf_m.trial_subset]
print(tone_list)

tf_m.w_CD = 0.51
tf_m.w_CI = 0.5
tf_m.beta = 1.5 
tf_m.obs_var1 = 0
test_data = tf_m.test_agent(sim_data)

f_size = [12,10]
n_bins = 14
s_noise = 0
f_control = tf_a.grid_plot_test(f_size, '2C3A-CONTROL', test_data, param, tf_m, init_f, 11, s_noise, tf_m.trial_subset)
print('Performance: ', tf_a.get_perf(test_data), ' // P(Break): ', tf_a.get_pbreak(test_data, tf_m))

#%%

# OPTOGENETICS EXPERIMENTS 

# DLS perturbation 
pert_list_a2a = [0.94, 0.93, 0.92]

tf_m.perturbation_type = 'iDLS-BL'
metric_a2a_bl = []
pert_a2a_bl = []
for i,e in enumerate(pert_list_a2a):
    tf_m.pert_mag = e
    pert_a2a_bl.append([tf_m.test_agent(sim_data), e])
    metric_a2a_bl.append(tf_a.get_behavior_metrics_opto(test_data, pert_a2a_bl[i][0], tf_m))
print(metric_a2a_bl)

tf_m.perturbation_type = 'iDLS-CS'
metric_a2a_cs = []
pert_a2a_cs = []
for i,e in enumerate(pert_list_a2a):
    tf_m.pert_mag = e
    pert_a2a_cs.append([tf_m.test_agent(sim_data), e])
    metric_a2a_cs.append(tf_a.get_behavior_metrics_opto(test_data, pert_a2a_cs[i][0], tf_m))
print(metric_a2a_cs)

tf_m.perturbation_type = 'iDLS-CL'
metric_a2a_cl = []
pert_a2a_cl = []
for i,e in enumerate(pert_list_a2a):
    tf_m.pert_mag = e
    pert_a2a_cl.append([tf_m.test_agent(sim_data), e])
    metric_a2a_cl.append(tf_a.get_behavior_metrics_opto(test_data, pert_a2a_cl[i][0], tf_m))
print(metric_a2a_cl)

f1_a2a = []
for i in range(len(pert_list_a2a)):
    f1_a2a.append(tf_a.grid_plot_pert(f_size, '2C3A-Test-A2A_BL', pert_a2a_bl[i][0], param, tf_m, init_f, n_bins, pert_a2a_bl[i][1], tf_m.trial_subset))
    
f2_a2a = []
for i in range(len(pert_list_a2a)):
    f2_a2a.append(tf_a.grid_plot_pert(f_size, '2C3A-Test-A2A_CS', pert_a2a_cs[i][0], param, tf_m, init_f, n_bins, pert_a2a_cs[i][1], tf_m.trial_subset))
    
f3_a2a = []
for i in range(len(pert_list_a2a)):
    f3_a2a.append(tf_a.grid_plot_pert(f_size, '2C3A-Test-A2A_CL', pert_a2a_cl[i][0], param, tf_m, init_f, n_bins, pert_a2a_cl[i][1], tf_m.trial_subset))


#%%
# DMS perturbation

tf_m.perturbation_type = 'dDMS-BL'
pert_dm_bl = []
metric_dm_bl = []
for i,e in enumerate(pert_list_dm):
    tf_m.pert_mag = e
    pert_dm_bl.append([tf_m.test_agent(sim_data,), e])
    metric_dm_bl.append(tf_a.get_behavior_metrics_opto(test_data, pert_dm_bl[i][0], tf_m))
print(metric_dm_bl)

tf_m.perturbation_type = 'dDMS-CS'
metric_dm_cs = []
pert_dm_cs = []
for i,e in enumerate(pert_list_dm):
    tf_m.pert_mag = e
    pert_dm_cs.append([tf_m.test_agent(sim_data,), e])
    metric_dm_cs.append(tf_a.get_behavior_metrics_opto(test_data, pert_dm_cs[i][0], tf_m))
print(metric_dm_cs)

tf_m.perturbation_type = 'dDMS-CL'
metric_dm_cl = []
pert_dm_cl = []
for i,e in enumerate(pert_list_dm):
    tf_m.pert_mag = e
    pert_dm_cl.append([tf_m.test_agent(sim_data), e])
    metric_dm_cl.append(tf_a.get_behavior_metrics_opto(test_data, pert_dm_cl[i][0], tf_m))
print(metric_dm_cl)

f1_dm = []
for i in range(len(pert_list_dm)):
    f1_dm.append(tf_a.grid_plot_pert(f_size, '2C3A-Test-DM_BL', pert_dm_bl[i][0], param, tf_m, init_f, n_bins, pert_dm_bl[i][1], tf_m.trial_subset))
    
f2_dm = []
for i in range(len(pert_list_dm)):
    f2_dm.append(tf_a.grid_plot_pert(f_size, '2C3A-Test-DM_CS', pert_dm_cs[i][0], param, tf_m, init_f, n_bins, pert_dm_cs[i][1], tf_m.trial_subset))
    
f3_dm = []
for i in range(len(pert_list_dm)):
    f3_dm.append(tf_a.grid_plot_pert(f_size, '2C3A-Test-DM_CL', pert_dm_cl[i][0], param, tf_m, init_f, n_bins, pert_dm_cl[i][1], tf_m.trial_subset))



#%%
# PERCEPTUAL POLICY

def conv_gauss(arr, sigma):
    size = int(2 * np.ceil(2 * sigma) + 1)
    x = np.linspace(-size / 2, size / 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    kernel /= np.sum(kernel)
    convolved = np.convolve(arr, kernel, mode='same')
    return convolved

def test_agent_act(tf_m, Act):
        
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
        env = tf_m.env
        
        # successor matrices
        
        M0 = np.zeros((tf_m.env.n_total_states, tf_m.env.n_total_states, 3))
        M1 = np.zeros((3, tf_m.env.n_total_states, tf_m.env.n_total_states))
            
        # behavioural measures
        choices = np.zeros(tf_m.n_test_eps) # ???
        behaviour_h = (-1) * np.ones((tf_m.n_test_episodes, env.n_total_states)) #episode_number, trial_id, action // initialised at -1
        choice_v = np.zeros((tf_m.n_test_episodes, env.n_total_states)) #episode_number, trial_id - (0,1,2,3) - (hodl, premature, correct, incorrect)
        trial_h = np.zeros((tf_m.n_test_episodes,2))
        rwd_hist = np.zeros((tf_m.n_test_episodes, env.n_total_states))
        state_visits = [np.zeros((2,env.n_total_states)), np.zeros((env.n_total_states,tf_m.n_actions))]
        
        state_data = []
        
        T = np.zeros((env.n_total_states, env.n_total_states))
        delta_spe = 0
        eta = 0.1
        
        Te = np.zeros((env.n_total_states, env.n_total_states))
        delta_spe_e = 0
        etr = np.zeros((env.n_total_states))
        gamma_e = 0.98
        lamb_e = 1
        eta_e = 1

        if tf_m.perturbation_type:
            print(tf_m.perturbation_type)

        print('running episodes')

        for episode in range(tf_m.n_test_episodes):

            if episode == int(tf_m.n_test_episodes/3):
                print('30%')

            if episode == int(tf_m.n_test_episodes/2):
                print('50%')
            
            if episode == int(3 * tf_m.n_test_episodes/4):
                print('75%')

            # saving trial history
            # randomly sampled trials no biasing
            current_tone = np.random.choice(tf_m.trial_subset)

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
            
            tcm_v = np.zeros((env.n_total_states, env.n_actions))

            for t in range(env.n_states):
                
                # POLICY -------------------------------------------------------------------
                
                action = tf_m.p_policy(Act[current_state, :], tf_m.beta)

                # ENVIRONMENT STATE UPDATE --------------------------------------------------
                
                next_state, next_state2, reward, choice_type, terminal_flag = env.get_outcome(trial_id, current_state, current_state2, action)
                
                # BEHAVIOUR HISTORY----------------------------------------------------------
                
                behaviour_h[episode, current_state] = action
                rwd_hist[episode, current_state] = reward
                choice_v[episode, current_state] = choice_type
                
                ### STATE VISIT UPDATE

                state_visits[0][0][current_state] += 1
                
                ### SUCCESSOR MATRICES -------------------------------------------------------
                
                # sucessor without discount
                M0[current_obs, next_obs, action] += 1
                
                next_action = tf_m.p_policy(Act[next_state, :], tf_m.beta)
                I = tf_m.onehot(current_state, tf_m.env.n_total_states)
                td_error = (I + tf_m.gamma_td * M1[next_action, next_state, :] - M1[action, current_state, :])
                M1[action, current_state, :] += tf_m.alpha_td * td_error
                
                ### FORWARD MODEL ----------------------------------------------------------
                
                delta_spe = 1 - T[current_state, next_state]
                T[current_state, next_state] += eta * delta_spe
                T[current_state, np.arange(T.shape[0]) != next_state] *= (1-eta)
                
                ### FORWARD MODEL w/ eligibility traces ------------------------------------------
                
                delta_spe_e = 1 - Te[current_state, next_state]
                
                etr[current_state] = etr[current_state] * gamma_e * lamb_e + 1
                etr[np.arange(etr.shape[0]) != current_state] = etr[np.arange(etr.shape[0]) != current_state] * gamma_e * lamb_e
                
                Te[current_state, next_state] += eta_e * delta_spe_e * etr[current_state]
                Te[current_state, np.arange(Te.shape[0]) != next_state] *= (1 - eta_e * etr[current_state])

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
                
        data_dict = {'n_episodes' : tf_m.n_test_episodes, 'choices': choices, 'trial_h' : trial_h, 
                    'behaviour_h' : behaviour_h, 'rwd_hist':rwd_hist, 
                    'state_visits_full' : state_visits, 'state_data': state_data,
                    'trial_h':trial_h, 'pert_type' :tf_m.perturbation_type,
                    'pert_mag' : tf_m.pert_mag, 'param' : tf_m.param, 'choice_v' : choice_v,
                    'M0': M0, 'M1' : M1, 'T' : T, 'Te' : Te}

        return data_dict

# fix this code, there's replication of variables - n_test_eps

A_DLS = copy.copy(sim_data['A'])
Act_Opt = A_DLS[0,:,:] - A_DLS[1,:,:]


tf_m.n_test_episodes = 10000
tf_m.perturbation_type = False
tf_m.trial_subset = [3,5,6,7,8,9,10,11,12] # this should be tone subset not trial subset
tone_list = [param['second_tone_list'][i] for i in tf_m.trial_subset]
print(tone_list)
tf_m.obs_var1 = 0
test_data_act = test_agent_act(tf_m, Act_Opt)

T_test = test_data_act['T']
plt.imshow(1-T_test[0:40, :], cmap = 'gray')
plt.xlabel('All next states')
plt.ylabel('pre 2nd tone states')
plt.savefig("transition_matrix.pdf")
plt.show()

M_sr_t = test_data['M1'][0,:,:]+test_data['M1'][1,:,:]+test_data['M1'][2,:,:]
plt.imshow(M_sr_t)

M1_act = test_data_act['M1']
plt.imshow(M1_act[2,:,:][0:40,:])
plt.show()
num_plots = M1_act[2, 0:40,40:80].shape[1]
cmap_name = 'cool'
cmap = plt.get_cmap(cmap_name)
colormap = cmap(np.linspace(0, 1, num_plots))


#%%

fig = plt.figure(figsize = [7,3])
T_conv = []
for i,e in enumerate(T_test[0:40,:]):
    plt.plot(conv_gauss(e,2), color = colormap[i])
    T_conv.append(conv_gauss(e,2))

T_conv = np.asarray(T_conv)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('States')
plt.ylabel(r'M(s,s)')
plt.savefig("transitions_all.pdf")
plt.show()


# fix this code, there's replication of variables - n_test_eps
tf_m.n_test_episodes = 10000
tf_m.perturbation_type = False
tf_m.trial_subset = [3,5,6,7,8,9,10,11,12] # this should be tone subset not trial subset
tone_list = [param['second_tone_list'][i] for i in tf_m.trial_subset]
print(tone_list)
tf_m.obs_var1 = 0
test_data_act = test_agent_act(tf_m, Act_Opt)


T_test = test_data_act['T']
plt.imshow(1-T_test[0:40, :], cmap = 'gray')
plt.xlabel('All next states')
plt.ylabel('pre 2nd tone states')
plt.savefig("transition_matrix.pdf")
plt.show()

fig = plt.figure(figsize = [7,3])
T_conv = []
for i,e in enumerate(T_test[0:40,:]):
    plt.plot(conv_gauss(e,2), color = colormap[i])
    T_conv.append(conv_gauss(e,2))

T_conv = np.asarray(T_conv)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('States')
plt.ylabel(r'M(s,s)')
plt.savefig("transitions_all.pdf")
plt.show()


#%%
# NMF DECOMPOSITION

from sklearn.decomposition import NMF

n_comp = 2
model = NMF(n_components=n_comp, init='random', random_state=0)
W = model.fit_transform(T_conv.T)
H = model.components_

fig = plt.figure(figsize = [7,3])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('States(s)')
plt.ylabel(r'NMF Components(s)')
plt.plot(H[0,:], color = 'green')
plt.plot(H[1,:], color = 'purple')
plt.savefig("nmf_components.pdf")
plt.show()


fig = plt.figure(figsize = [7,3])
for i,e in enumerate(T_test[0:40,40:80]):
    plt.plot(conv_gauss(e,2), color = colormap[i])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('States')
plt.ylabel(r'$\rho(z|s)$')
plt.title('Post 2nd tone predictions')
labels = [e for e in np.linspace(40,80,num = 9,dtype = int)]
pos = np.linspace(0,40,num = 9)
plt.xticks(pos, labels)
plt.savefig("state_probabilities.pdf")

## 

# SAVING DATA 


# %%
