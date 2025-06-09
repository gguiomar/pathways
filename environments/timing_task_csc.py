import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy


class timing_task_csc_v8():
    
    def __init__(self, n_states, second_tone_list, reward_magnitude, 
                 reward_probability, punishment_magnitude, punishment_probability, dwell_time_list):
        
        # Task variables
        self.n_states = n_states # separate total number of transitions from total number of states
        self.n_total_states = 2 * n_states 
        self.n_actions = 3
        
        # trial information
        self.action_dict = np.arange(self.n_actions)
        self.second_tone_list = second_tone_list
        self.trial_types = len(self.second_tone_list)
        self.n_trials = len(self.second_tone_list) # ambivalence in this definition
        self.trials = np.zeros((self.n_trials, self.n_states*2, self.n_states))
        self.tone_boundary = 1.5
        self.jump_correction = 2
        self.jump_magnitude = self.n_states
        
        # n_trials_per_tone, max_bs, min_bs, second_tones, time
        self.n_tones = len(self.second_tone_list)
        self.ti = 0.0 # trial init (in seconds)
        self.tf = 2.5 # trial max time (in seconds)

        # new formulation with box sizes
        # the sizes and number of trials come in one single vector now
        self.b_sizes = dwell_time_list # all the dwell times in (ms) that the environment is going to generate
        self.n_trials_per_tone = dwell_time_list.shape[0]
        
        # tried 0.0001 but this shit blows up
        #self.div = (self.tf-self.ti)/0.001 # number of divisions of the temporal variable (in ms)
        self.tic = 0.001
        self.div = (self.tf-self.ti)/self.tic # number of divisions of the temporal variable (in ms)
        self.time = np.linspace(self.ti, self.tf, int(self.div))
        
        # Reward variables
        self.reward_state = [0,0]
        self.reward_magnitude = reward_magnitude
        self.punishment_magnitude = punishment_magnitude
        self.reward_probability = reward_probability
        self.punishment_probability = punishment_probability

    
    """
    GENERATORS
    These functions generate the necessary ingredients to define the Markov Decision Process (MDP)
    of the interval timing categorisation. 

    ---- Short description of the task ---
    ...
    """

    def print_env_version(self):
        print('This is environment V8')

    def generate_environment(self):
        """
        Main generating function: Takes the state representation generates the state sequence.
        """
        self.get_time_trajectories()
        self.generate_state_sequence()
        self.generate_environment_solution()       
        
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def get_time_trajectories(self):
        # sharing trajectories amongst the two observers so that we get the same 
        # trial to trial variability 
        
        print('generating temporal trajectories...')
        # generates a trajectory for a given tone
        zt = 0
        zt2 = 0
        z_vec = np.zeros((self.n_tones, self.n_trials_per_tone, 2 ,int(self.div)))
        z_vec2 = np.zeros((self.n_tones, self.n_trials_per_tone, 2 ,int(self.div)))
        b_size_vec = np.zeros((self.n_tones, self.n_trials_per_tone))

        # list of all the available box sizes
        add_tic = False
        c = 0
        for i_t, tr_tone in enumerate(self.second_tone_list):
            for tr in range(self.n_trials_per_tone):
                
                # define a list of trials that repeats at each tone
                b_size = self.b_sizes[tr] 
                b_size_vec[i_t, tr] = b_size
                zt = 0
                zt2 = 0
                for e, t in enumerate(self.time):
                    if t == self.find_nearest(self.time, tr_tone)[0]:
                        zt += self.jump_magnitude - self.jump_correction
                    if np.mod(e, b_size) == b_size - 1:
                        zt += 1
                        zt2 += 1

                    # jumping observation model
                    z_vec[i_t,tr,0,e] = t
                    z_vec[i_t,tr,1,e] = zt

                    # non-jumping obs model
                    z_vec2[i_t,tr,0,e] = t
                    z_vec2[i_t,tr,1,e] = zt2
                    

        self.z_vec = z_vec
        self.z_vec2 = z_vec2
        self.b_size_vec = b_size_vec
    
    def generate_state_sequence(self): 
        
        print('generating state sequences for d1d2...')
        all_traj = np.zeros((self.n_tones, self.n_trials_per_tone, self.n_states * 2)) 
        for tone in range(self.n_tones):
            for trial in range(self.n_trials_per_tone):
                l_traj = np.unique(self.z_vec[tone, trial, 1, :]).astype(int)
                for i,e in enumerate(l_traj):
                    if l_traj[i] < self.n_states * 2:
                        all_traj[tone, trial, i] = l_traj[i]
        all_traj = all_traj.astype(int)
        # matrix with all trajectories
        self.trial_st = all_traj
        
        # same but for the no-second tone jump
        print('generating state sequences for third observer...')
        all_traj2 = np.zeros((self.n_tones, self.n_trials_per_tone, self.n_states * 2)) 
        for tone in range(self.n_tones):
            for trial in range(self.n_trials_per_tone):
                l_traj2 = np.unique(self.z_vec2[tone, trial, 1, :]).astype(int)
                for i,e in enumerate(l_traj2):
                    if l_traj2[i] < self.n_states * 2:
                        all_traj2[tone, trial, i] = l_traj2[i]
        all_traj2 = all_traj2.astype(int)
        self.trial_st2 = all_traj2
    
    def generate_environment_solution(self):
        print('generating environment solutions')
        # generates the policy for the optimal agent - use state index to match actions and states
        all_sol = 2 * np.ones((self.n_tones, self.n_trials_per_tone, self.n_states * 2))
        for tone in range(self.n_tones):
            tone_s = self.second_tone_list[tone]
            for trial in range(self.n_trials_per_tone):
                for i,e in enumerate(self.trial_st[tone, trial,:]):
                    if e >= self.n_states/2: # rewards are in the upper half of states
                        if tone_s < self.tone_boundary: # left and right boundary conditions
                            all_sol[tone,trial,i] = 0 # left action is the correct one
                        else: 
                            all_sol[tone,trial,i] = 1 # right action is the correct one
        
        # matrix with the solutions for each trial
        self.opt_act = all_sol.astype(int)      

    """
    VISUALIZATIONS
    These functions show human readable representations of the state trajectories
    """
    def plot_zvec(self, z_vec, tones):
    
        f, ax = plt.subplots(1,1,figsize=(7,5))
        for tone in tones:
            for e in z_vec[tone,:,:,:]:
                ax.plot(e[0], e[1])
                ax.axhline(self.n_states, color = 'k', linewidth = 0.5)
                ax.axhline(self.n_states/2, color = 'k', linewidth = 0.5)
                ax.axhline(self.n_states + self.n_states/2 , color = 'k', linewidth = 0.5)
                ax.axvline(0.6, color = 'k', linewidth = 0.5)
                ax.axvline(1.5, color = 'k', linewidth = 0.5)
                ax.axvline(2.4, color = 'k', linewidth = 0.5)
                ax.set_ylim(0, self.n_states *2)
                ax.set_xlim(self.ti, self.tf)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('State')
            ax.plot(self.time, np.linspace(0,self.n_states,self.div), linewidth = 0.5, color = 'r')

        return f
        
    def plot_env_trajectories(self, f_size, z_vec):

        viridis = cm.get_cmap('jet', 12)
        cmap_list = viridis(np.linspace(0, 1, 100))
        cmap_shift = 20

        f, ax = plt.subplots(1,1,figsize=(f_size[0],f_size[1]))
        for tone in range(self.n_tones):
            clr = cmap_list[tone+10]
            for e in z_vec[tone,:,:,:]:
                ax.plot(e[0], e[1], color = clr)
                ax.axvline(0.6, color = 'k', linewidth = 0.5)
                ax.axvline(1.5, color = 'k', linewidth = 0.5)
                ax.axvline(2.4, color = 'k', linewidth = 0.5)
                ax.axhline(self.n_states, color = 'k', linewidth = 0.5)
                ax.set_ylim(0,self.n_total_states)
                ax.set_xlim(self.ti, self.tf)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('State')

    """
    TESTING THE ENVIRONMENT
    These functions go through all states and actions in order to check for
    weird state,action transtions
    """         
    def test_environment_with_solution(self, tone):
        # testing environment with correct actions
        cs = 0
        ns = 0
        for trial in range(self.n_trials_per_tone):
            print('---- Tone, Trial', (tone, trial))
            for i in range(self.n_total_states):
                state_index = np.argwhere(self.trial_st[tone, trial] == cs)[0][0]
                action = self.opt_act[tone, trial, state_index]
                ns, r, c, tf = self.get_outcome([tone, trial], cs, action)
                print(('action', action), (cs, ns, r, c, tf))
                cs = ns
                if tf == 1:
                    cs = 0
                    ns = 0
                    break
                

    def test_environment_all_variables(self, actions):
        # testing environment with all actions, tones, trials and states
        for action in actions:
            print('Action', action)
            for tone in range(self.n_tones):
                print('Tone', tone)
                for trial in range(self.n_trials_per_tone):
                    print('Trial', trial)
                    for st in range(self.n_total_states):
                        print((tone, trial, action, st), self.get_outcome([tone, trial], st, st, action))
    """
    GET OUTCOME
    Main function of the enviromment; defines the MDP
    trial_id - contains both trial_type (tone identity) and trial index - UPDATE 
    """
    
    # # inserting more actions
    def get_outcome(self, trial_id, current_state, current_state2, action):
            
        next_state = 0
        next_state2 = 0
        reward = 0
        
        check_valid_state = np.argwhere(self.trial_st[trial_id[0], trial_id[1]] == current_state).shape[0] # should be > 0
        choice_type = 0 # 0,1,2,3 - normal, premature, correct, incorrect
        terminal_flag = 0
        
        if check_valid_state:
        
            state_index = np.argwhere(self.trial_st[trial_id[0], trial_id[1]] == current_state)[0][0]

            # when we reach the final terminal state we go back to the initial state
            if current_state == self.n_total_states - 1:
                reward = 0
                next_state = 0
                terminal_flag = 1

            else:    
                # MAKING A CHOICE AND GOING TO TERMINAL STATE
                if action != 2:

                    # if a decision is made on second tone states
                    if current_state > self.n_states: # buffer states 

                        # CORRECT ACTION
                        if action == self.opt_act[trial_id[0], trial_id[1], state_index]:
                            reward = self.reward_magnitude
                            next_state = self.n_total_states - 1 # transition into terminal states
                            choice_type = 2 # CORRECT ACTION
                            terminal_flag = 1

                        # INCORRECT ACTION
                        else:
                            reward = self.punishment_magnitude
                            next_state = self.n_total_states - 1 # transition into terminal states
                            terminal_flag = 1
                            choice_type = 3 # INCORRECT ACTION

                    # if decision is made on pre-second tone states
                    if current_state <= self.n_states:

                        reward = self.punishment_magnitude
                        next_state = self.n_total_states - 1 # transition into terminal states
                        terminal_flag = 1
                        choice_type = 1 # PREMATURE ACTION

                # HODL
                if action == 2:
                    #print('Holding')
                    state_index = np.argwhere(self.trial_st[trial_id[0], trial_id[1]] == current_state)[0][0] #current state index
                    if state_index < np.argmax(self.trial_st[trial_id[0], trial_id[1]]):
                        next_state = self.trial_st[trial_id[0], trial_id[1]][state_index + 1]
                        terminal_flag = 0
                        choice_type = 0 # HODL
                    else:
                        next_state = 0
                        reward = 0
                        choice_type = 0 # HODL
                    
                    # SECOND OBSERVER STATE TRANSITIONS
                    state_index2 = np.argwhere(self.trial_st2[trial_id[0], trial_id[1]] == current_state2)[0][0] #current state index
                    if state_index2 < np.argmax(self.trial_st2[trial_id[0], trial_id[1]]):
                        next_state2 = self.trial_st2[trial_id[0], trial_id[1]][state_index2 + 1]
                    else:
                        next_state2 = 0
        else:
            next_state = 0
            next_state2 = 0
            reward = 0
                
                
        return next_state, next_state2, reward, choice_type, terminal_flag
