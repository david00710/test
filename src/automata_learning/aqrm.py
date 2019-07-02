import numpy as np
import random, time
import tensorflow as tf
from worlds.game import *
from automata_learning.policy_bank_dqn import PolicyBankDQN
from common.schedules import LinearSchedule
from common.replay_buffer import create_experience_replay_buffer
from automata_learning.Traces import Traces
from tester.saver import Saver
from reward_machines.reward_machine import RewardMachine

#import pdb




def run_aqrm_task(sess, environment_rm_file, learned_rm_file, policy_bank, tester_true, tester_learned, curriculum, replay_buffer, beta_schedule, show_print):
    """
    This code runs one training episode. 
        - rm_file: It is the path towards the RM machine to solve on this episode
        - environment_rm: an environment reward machine, the "true" one, underlying the execution
    """
    # Initializing parameters and the game
    learning_params = tester_learned.learning_params
    testing_params = tester_learned.testing_params

    """
     here, tester holds all the machines. we would like to dynamically update the machines every so often.
     an option might be to read it every time a new machine is learnt
     """
    reward_machines = tester_learned.get_reward_machines()

    task_params = tester_learned.get_task_params(learned_rm_file) # rm_files redundant here unless in water world (in which case it provides the map files based on the task)
    rm_true = tester_true.get_reward_machines()[0] #add one more input n to track tasks at hand, replace 0 with n
    rm_learned = tester_learned.get_reward_machines()[0] #ditto

    task = Game(task_params)
    actions = task.get_actions()
    #pdb.set_trace()
    num_features = len(task.get_features())
    num_steps = learning_params.max_timesteps_per_task
    
    training_reward = 0
    # Getting the initial state of the environment and the reward machine
    s1, s1_features = task.get_state_and_features()
    u1 = rm_learned.get_initial_state()

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps)
    all_events = []
    for t in range(num_steps):

        # Choosing an action to perform
        if random.random() < 0.1:
            a = random.choice(actions)
        else:
            #IG: current problem: there is no machine so  a default behavior is to stop the exploration. We would, however, like to explore (randomly if necessary).
            # how to accomplish that?

            #if using suggestions in comments on line 33, replace 0 with n
            a = policy_bank.get_best_action(0, u1, s1_features.reshape((1,num_features)))

        # updating the curriculum
        curriculum.add_step()
                
        # Executing the action
        task.execute_action(a)
        a = task.get_last_action() # due to MDP slip
        s2, s2_features = task.get_state_and_features()
        events = task.get_true_propositions()
        all_events.append(events)


        u2 = rm_learned.get_next_state(u1, events)
        reward = rm_true.get_reward(u1,u2,s1,a,s2)


        training_reward += reward
        
        # Getting rewards and next states for each reward machine
        rewards, next_states = [],[]
        for j in range(len(reward_machines)):
            j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
            rewards.append(j_rewards)
            next_states.append(j_next_states)
        # Mapping rewards and next states to specific policies in the policy bank
        rewards = policy_bank.select_rewards(rewards)
        next_policies = policy_bank.select_next_policies(next_states)

        # Adding this experience to the experience replay buffer
        replay_buffer.add(s1_features, a, s2_features, rewards, next_policies)

        # Learning
        if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.train_freq == 0:
            if learning_params.prioritized_replay:
                experience = replay_buffer.sample(learning_params.batch_size, beta=beta_schedule.value(curriculum.get_current_step()))
                S1, A, S2, Rs, NPs, weights, batch_idxes = experience
            else:
                S1, A, S2, Rs, NPs = replay_buffer.sample(learning_params.batch_size)
                weights, batch_idxes = None, None
            abs_td_errors = policy_bank.learn(S1, A, S2, Rs, NPs, weights) # returns the absolute td_error
            if learning_params.prioritized_replay:
                new_priorities = abs_td_errors + learning_params.prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)
            
        # Updating the target network
        if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.target_network_update_freq == 0:
            policy_bank.update_target_network()

        # Printing
        if show_print and (t+1) % learning_params.print_freq == 0:
            print("Step:", t+1, "\tTotal reward:", training_reward)

        # Testing
        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
            tester_true.run_test(curriculum.get_current_step(), sess, run_aqrm_test, policy_bank, num_features)

        # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm_learned.is_terminal_state(u2):
            # Restarting the game
            task = Game(task_params)
            s2, s2_features = task.get_state_and_features()
            u2 = rm_learned.get_initial_state()

            if curriculum.stop_task(t):
                break
        
        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

    if show_print: print("Done! Total reward:", training_reward)
    return all_events, training_reward


def run_aqrm_test(sess, reward_machines, task_params, task_rm_id, learning_params, testing_params, policy_bank, num_features):
    # Initializing parameters
    task = Game(task_params)
    rm = reward_machines[task_rm_id]
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Choosing an action using the right policy
        a = policy_bank.get_best_action(task_rm_id, u1, s1_features.reshape((1,num_features)), add_noise=False)

        # Executing the action
        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        u2 = rm.get_next_state(u1, task.get_true_propositions())
        r = rm.get_reward(u1,u2,s1,a,s2)

        r_total += r * learning_params.gamma**t
        
        # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm.is_terminal_state(u2):
            break
        
        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2
    
    return r_total

def run_aqrm_experiments(alg_name, tester, tester_learned, curriculum, num_times, show_print):
    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params


    # Running the tasks 'num_times'
    time_init = time.time()
    for t in range(num_times):
        # Setting the random seed to 't'

        random.seed(t)
        sess = tf.Session()

        # Reseting default values
        curriculum.restart()

        # Creating the experience replay buffer
        replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size, learning_params.prioritized_replay, learning_params.prioritized_replay_alpha, learning_params.prioritized_replay_beta0, curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions  = len(task_aux.get_actions())

        hypothesis_machine = tester.get_hypothesis_machine()
        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, hypothesis_machine)
        all_traces = Traces()
        # Task loop
        num_episodes = 0
        while not curriculum.stop_learning():
            num_episodes += 1
            if len(all_traces.positive) > 0:
                print(all_traces)
                #pdb.set_trace()
            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            #underlying_rm_file = curriculum.get_next_task()
            rm_file_truth = '../experiments/office/reward_machines/t1.txt' #set file path at beginning
            rm_file_learned = '../experiments/office/reward_machines/xyz.txt' #this txt file should be updated in learning section, where run_aqrm_task can be repeated?
            # Running 'task_rm_id' for one episode
            all_events, found_reward = run_aqrm_task(sess, rm_file_truth, rm_file_learned, policy_bank, tester, tester_learned, curriculum, replay_buffer, beta_schedule, show_print)
            

            expected_reward = hypothesis_machine.calculate_reward(all_events)
            #pdb.set_trace()
            if not found_reward == expected_reward:
                # the learning should happen here
                print("learning")


            # save the trace, it will be used to create an underlying reward automaton
            all_traces.add_trace(all_events, found_reward)

        tf.reset_default_graph()
        sess.close()
        
        # Backing up the results
        saver.save_results()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")
