"""
Perceptual policy utility functions.
Contains analysis functions and plotting utilities for perceptual policy analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy
from sklearn.decomposition import NMF

# Local imports
from utils import conv_gauss

def test_agent_act(tf_m, Act):
    """
    Test agent with specific action preferences.
    
    Args:
        tf_m: Trained agent
        Act: Action preferences matrix
        
    Returns:
        dict: Test data with action preferences
    """
    # States and Observations
    current_state = 0
    next_state = 0
    
    previous_obs = 0
    current_obs = 0
    next_obs = 0
    
    current_obs2 = 0
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
    choices = np.zeros(tf_m.n_test_eps)
    behaviour_h = (-1) * np.ones((tf_m.n_test_episodes, env.n_total_states))
    choice_v = np.zeros((tf_m.n_test_episodes, env.n_total_states))
    trial_h = np.zeros((tf_m.n_test_episodes, 2))
    rwd_hist = np.zeros((tf_m.n_test_episodes, env.n_total_states))
    state_visits = [np.zeros((2, env.n_total_states)), np.zeros((env.n_total_states, tf_m.n_actions))]
    
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
            
    data_dict = {'n_episodes': tf_m.n_test_episodes, 'choices': choices, 'trial_h': trial_h, 
                'behaviour_h': behaviour_h, 'rwd_hist': rwd_hist, 
                'state_visits_full': state_visits, 'state_data': state_data,
                'trial_h': trial_h, 'pert_type': tf_m.perturbation_type,
                'pert_mag': tf_m.pert_mag, 'param': tf_m.param, 'choice_v': choice_v,
                'M0': M0, 'M1': M1, 'T': T, 'Te': Te}

    return data_dict

def plot_transition_matrix(T_test, save_plots=True):
    """
    Plot transition matrix.
    
    Args:
        T_test: Transition matrix
        save_plots: Whether to save plots
        
    Returns:
        matplotlib.figure.Figure: Transition matrix plot
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(1-T_test[0:40, :], cmap='gray')
    ax.set_xlabel('All next states')
    ax.set_ylabel('pre 2nd tone states')
    
    if save_plots:
        fig.savefig("plots/transition_matrix.png", bbox_inches='tight', dpi=300)
        print("Transition matrix saved to plots/transition_matrix.png")
    
    return fig

def plot_transitions_all(T_test, save_plots=True):
    """
    Plot all transitions with Gaussian convolution.
    
    Args:
        T_test: Transition matrix
        save_plots: Whether to save plots
        
    Returns:
        tuple: (figure, convolved transitions)
    """
    fig = plt.figure(figsize=[7, 3])
    T_conv = []
    
    num_plots = T_test[0:40, :].shape[0]
    cmap_name = 'cool'
    cmap = plt.get_cmap(cmap_name)
    colormap = cmap(np.linspace(0, 1, num_plots))
    
    for i, e in enumerate(T_test[0:40, :]):
        conv_result = conv_gauss(e, 2)
        plt.plot(conv_result, color=colormap[i])
        T_conv.append(conv_result)

    T_conv = np.asarray(T_conv)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('States')
    plt.ylabel(r'M(s,s)')
    
    if save_plots:
        fig.savefig("plots/transitions_all.png", bbox_inches='tight', dpi=300)
        print("All transitions plot saved to plots/transitions_all.png")
    
    return fig, T_conv

def perform_nmf_decomposition(T_conv, n_comp=2, save_plots=True):
    """
    Perform NMF decomposition on transition data.
    
    Args:
        T_conv: Convolved transition matrix
        n_comp: Number of components
        save_plots: Whether to save plots
        
    Returns:
        tuple: (figure, W, H)
    """
    model = NMF(n_components=n_comp, init='random', random_state=0)
    W = model.fit_transform(T_conv.T)
    H = model.components_

    fig = plt.figure(figsize=[7, 3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('States(s)')
    plt.ylabel(r'NMF Components(s)')
    plt.plot(H[0, :], color='green', label='Component 1')
    plt.plot(H[1, :], color='purple', label='Component 2')
    plt.legend()
    
    if save_plots:
        fig.savefig("plots/nmf_components.png", bbox_inches='tight', dpi=300)
        print("NMF components plot saved to plots/nmf_components.png")
    
    return fig, W, H

def plot_state_probabilities(T_test, save_plots=True):
    """
    Plot state probabilities for post 2nd tone predictions.
    
    Args:
        T_test: Transition matrix
        save_plots: Whether to save plots
        
    Returns:
        matplotlib.figure.Figure: State probabilities plot
    """
    num_plots = T_test[0:40, 40:80].shape[0]
    cmap_name = 'cool'
    cmap = plt.get_cmap(cmap_name)
    colormap = cmap(np.linspace(0, 1, num_plots))
    
    fig = plt.figure(figsize=[7, 3])
    for i, e in enumerate(T_test[0:40, 40:80]):
        plt.plot(conv_gauss(e, 2), color=colormap[i])
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('States')
    plt.ylabel(r'$\rho(z|s)$')
    plt.title('Post 2nd tone predictions')
    
    labels = [e for e in np.linspace(40, 80, num=9, dtype=int)]
    pos = np.linspace(0, 40, num=9)
    plt.xticks(pos, labels)
    
    if save_plots:
        fig.savefig("plots/state_probabilities.png", bbox_inches='tight', dpi=300)
        print("State probabilities plot saved to plots/state_probabilities.png")
    
    return fig

def create_perceptual_policy_grid_plot(T_test, save_plots=True):
    """
    Create a comprehensive grid plot for perceptual policy analysis.
    
    Args:
        T_test: Transition matrix
        save_plots: Whether to save plots
        
    Returns:
        matplotlib.figure.Figure: Grid plot figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Transition matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(1-T_test[0:40, :], cmap='gray')
    ax1.set_xlabel('All next states')
    ax1.set_ylabel('Pre 2nd tone states')
    ax1.set_title('Transition Matrix')
    
    # All transitions
    ax2 = fig.add_subplot(gs[0, 1])
    T_conv = []
    num_plots = T_test[0:40, :].shape[0]
    cmap_name = 'cool'
    cmap = plt.get_cmap(cmap_name)
    colormap = cmap(np.linspace(0, 1, num_plots))
    
    for i, e in enumerate(T_test[0:40, :]):
        conv_result = conv_gauss(e, 2)
        ax2.plot(conv_result, color=colormap[i])
        T_conv.append(conv_result)
    
    T_conv = np.asarray(T_conv)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('States')
    ax2.set_ylabel(r'M(s,s)')
    ax2.set_title('All Transitions')
    
    # NMF decomposition
    ax3 = fig.add_subplot(gs[0, 2])
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(T_conv.T)
    H = model.components_
    
    ax3.plot(H[0, :], color='green', label='Component 1')
    ax3.plot(H[1, :], color='purple', label='Component 2')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlabel('States')
    ax3.set_ylabel('NMF Components')
    ax3.set_title('NMF Decomposition')
    ax3.legend()
    
    # State probabilities
    ax4 = fig.add_subplot(gs[1, :])
    for i, e in enumerate(T_test[0:40, 40:80]):
        ax4.plot(conv_gauss(e, 2), color=colormap[i])
    
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_xlabel('States')
    ax4.set_ylabel(r'$\rho(z|s)$')
    ax4.set_title('Post 2nd tone predictions')
    
    labels = [e for e in np.linspace(40, 80, num=9, dtype=int)]
    pos = np.linspace(0, 40, num=9)
    ax4.set_xticks(pos)
    ax4.set_xticklabels(labels)
    
    plt.tight_layout()
    
    if save_plots:
        fig.savefig("plots/perceptual_policy_grid.png", bbox_inches='tight', dpi=300)
        print("Perceptual policy grid plot saved to plots/perceptual_policy_grid.png")
    
    return fig

def analyze_successor_representation(M1_act, save_plots=True):
    """
    Analyze and plot successor representation.
    
    Args:
        M1_act: Successor representation matrix
        save_plots: Whether to save plots
        
    Returns:
        matplotlib.figure.Figure: Successor representation plot
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M1_act[2, :, :][0:40, :])
    ax.set_xlabel('States')
    ax.set_ylabel('States')
    ax.set_title('Successor Representation')
    
    if save_plots:
        fig.savefig("plots/successor_representation.png", bbox_inches='tight', dpi=300)
        print("Successor representation plot saved to plots/successor_representation.png")
    
    return fig
