from __future__ import division
import numpy as np

class MC(object):
	def __init__(self, num_states, num_actions):
		self.Q = np.zeros((num_states,num_actions))
		self.policy_pi = np.random.randint(num_actions,size=num_states)
		self.action_reward_sum = np.zeros((num_states,num_actions))
		self.action_reward_count = np.zeros((num_states,num_actions))
		self.state_seen = []
	
	def update(self, sta_ind, act_ind, reward):
		# Accumulate these values used in computing statistics on this action value function Q^pi
		self.action_reward_count[sta_ind, act_ind] += 1
		self.action_reward_sum[sta_ind, act_ind] += reward
		self.Q[sta_ind, act_ind] = self.action_reward_sum[sta_ind, act_ind]/self.action_reward_count[sta_ind, act_ind] # take average
		
		self.policy_pi[sta_ind] = np.argmax(self.Q[sta_ind]) # greedy choice
	
	def record_state_seen(self, game_state):
		self.state_seen.append(np.array(game_state)) # [card_showing, largest, median, smallest]
	
	def clear_states_seen(self):
		self.state_seen = []
	
	def print_mc(self,run):
		np.savetxt('Q-%i.txt' % run, self.Q, fmt='%.8f')
		np.savetxt('policy_pi-%i.txt' % run, self.policy_pi, fmt='%i')
		np.savetxt('action_reward_count-%i.txt' % run, self.action_reward_count, fmt='%i')
		np.savetxt('action_reward_sum-%i.txt' % run, self.action_reward_sum, fmt='%i')