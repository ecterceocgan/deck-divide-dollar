from __future__ import division
import numpy as np
import time
import math
import mc # class for MC object

"""Monte Carlo agent learns to play divide-the-dollar."""

start = time.clock()

## Parameters for divide-the-dollar game ##
cards = [0.25, 0.50, 0.75] # specifies the unique cards in the deck: indexed as [0,1,2]
num_of_unique_cards = [16, 28, 16] # specifies the total number of each of the unique cards
num_cards = len(cards) # number of unique cards
deck_size = sum(num_of_unique_cards) # total number of cards in the deck
hand_size = 5 # number of cards in a player's hand
num_players = 2 # number of players
num_rounds = 1+(deck_size-(num_players*hand_size))//num_players # number of hands to play (until deck runs out)
num_episodes = 2000000 # number of full games to play

actions = {'small_spoil': 0, 'median': 1, 'large_max': 2}
num_actions = len(actions)

num_states = int((num_cards+1)*(math.factorial(3+num_cards-1))/(math.factorial(3)*math.factorial(num_cards-1)))
#true_state_index = np.loadtxt('true_state_index.txt')
true_state_index =  [0,1,2,-1,3,4,-1,-1,5,-1,-1,-1,-1,6,7,-1,-1,8,-1,-1,-1,-1,-1,-1,-1,-1,9,10,11,12,-1,13,14,-1,-1,15,-1,-1,-1,-1,16,17,-1,-1,18,-1,-1,-1,-1,-1,-1,-1,-1,19,20,21,22,-1,23,24,-1,-1,25,-1,-1,-1,-1,26,27,-1,-1,28,-1,-1,-1,-1,-1,-1,-1,-1,29,30,31,32,-1,33,34,-1,-1,35,-1,-1,-1,-1,36,37,-1,-1,38,-1,-1,-1,-1,-1,-1,-1,-1,39]

def init_mc():
	monte = mc.MC(num_states, num_actions)
	return monte

def load_deck():
	d = []
	for card, num in enumerate(num_of_unique_cards):
		d += [card]*num
	return np.array(d)

def play_action(card_showing, player, player_action):
	player_card_value = 0
	if card_showing == num_cards: # player's going first
		if player_action == actions['small_spoil']:
			player_card_value = cards[player[0]] # play smallest card
			card_showing = player[0] # update card showing
			player = np.delete(player, 0) # remove card from player's hand
		elif player_action == actions['large_max']:
			player_card_value = cards[player[-1]] # play largest card
			card_showing = player[-1] # update card showing
			player = np.delete(player, -1) # remove card from player's hand
		else:
			player_card_value = cards[player[hand_size//2]] # play median card
			card_showing = player[hand_size//2] # update card showing
			player = np.delete(player, hand_size//2) # remove card from player's hand
	else: # opponent went first, player's turn
		if player_action == actions['small_spoil']: # spoil with smallest card
			for c, pcard in enumerate(player):
				if cards[pcard] + cards[card_showing] > 1.0: # can spoil, play this card
					player_card_value = cards[player[c]]
					player = np.delete(player, c) # remove card from player's hand
					break
				elif c == len(player)-1: # can't spoil, play largest card
					player_card_value = cards[player[-1]]
					player = np.delete(player, -1) # remove card from player's hand
		elif player_action == actions['large_max']: # maximize score with largest card
			for c, pcard in enumerate(np.flipud(player)):
				if cards[pcard] + cards[card_showing] <= 1.0: # can maximize, play this card
					player_card_value = cards[player[len(player)-1-c]]
					player = np.delete(player, len(player)-1-c) # remove card from player's hand
					break
				elif len(player)-c == 0: # can't maximize, play smallest card
					player_card_value = cards[player[0]]
					player = np.delete(player, 0) # remove card from player's hand
		else:
			player_card_value = cards[player[hand_size//2]] # play median card
			player = np.delete(player, hand_size//2) # remove card from player's hand
	return card_showing, player, player_card_value

monte = init_mc()
opp_strategies = {'always-small_spoil': 0, 'always-median': 1, 'always-large_max': 2, 'random': 3}
opp_pol = opp_strategies['random']
#opp_pol = np.copy(monte.policy_pi) # self-play

wins = 0

fraction_won = open('fraction_won-%i.txt' % num_episodes, 'w') # used for plotting games won over time
score_every_hand = open('score_every_game-%i.txt' % num_episodes, 'w') # used for plotting score differential

for episode_index in xrange(num_episodes):
	# Load and shuffle deck
	deck = load_deck()
	np.random.shuffle(deck)
	
	mc_total_score = 0
	opp_total_score = 0
	
	# Deal initial hands
	mc_cards = np.sort(deck[:hand_size])
	deck = deck[hand_size:]
	opp_cards = np.sort(deck[:hand_size])
	deck = deck[hand_size:]
	
	monte.clear_states_seen()
	
	for round_index in xrange(num_rounds):
		mc_card_value = 0
		opp_card_value = 0
		
		# Determine the value of the card showing (0 if playing first; opponent's pick if playing second)
		card_showing = num_cards # an index of 'num_cards' corresponds to no card showing (i.e. zero)
		
		if round_index % 2 == 0:
			# MC player goes first
			mc_game_state = [card_showing, mc_cards[0], mc_cards[hand_size//2], mc_cards[-1]]
			monte.record_state_seen(mc_game_state)
			mc_policy_index = true_state_index[int(np.ravel_multi_index(mc_game_state, dims=(num_cards+1, num_cards, num_cards, num_cards)))]
			if round_index <= 1:
				monte.policy_pi[int(mc_policy_index)] = np.random.choice(actions.values()) # for exploring starts take an initial random policy
			mc_action = monte.policy_pi[int(mc_policy_index)]
			card_showing, mc_cards, mc_card_value = play_action(card_showing, mc_cards, mc_action)
			
			# Opponent goes second
			if opp_pol == opp_strategies['random']:
				opp_action = np.random.choice(actions.values()) # opponent strategy is to select random action
			else:
				opp_action = opp_pol # [0,1,or,2] execute opponent strategy
				#opp_action = opp_pol[true_state_index[int(np.ravel_multi_index([card_showing,opp_cards[0],opp_cards[hand_size//2],opp_cards[-1]], dims=(num_cards+1,num_cards,num_cards,num_cards)))]] # self-play
			card_showing, opp_cards, opp_card_value = play_action(card_showing, opp_cards, opp_action)
		else:
			# Opponent goes first
			if opp_pol == opp_strategies['random']:
				opp_action = np.random.choice(actions.values()) # opponent strategy is to select random action
			else:
				opp_action = opp_pol # [0,1,or,2] execute opponent strategy
				#opp_action = opp_pol[true_state_index[int(np.ravel_multi_index([card_showing,opp_cards[0],opp_cards[hand_size//2],opp_cardst[-1]], dims=(num_cards+1,num_cards,num_cards,num_cards)))]] # self-play
			card_showing, opp_cards, opp_card_value = play_action(card_showing, opp_cards, opp_action)
			
			# MC player goes second
			mc_game_state = [card_showing, mc_cards[0], mc_cards[hand_size//2], mc_cards[-1]]
			monte.record_state_seen(mc_game_state)
			mc_policy_index = true_state_index[int(np.ravel_multi_index(mc_game_state, dims=(num_cards+1, num_cards, num_cards, num_cards)))]
			if round_index <= 1:
				monte.policy_pi[int(mc_policy_index)] = np.random.choice(actions.values()) # for exploring starts take an initial random policy
			mc_action = monte.policy_pi[int(mc_policy_index)]
			card_showing, mc_cards, mc_card_value = play_action(card_showing, mc_cards, mc_action)
		
		# Determine score for playing this hand
		if mc_card_value + opp_card_value <= 1:
			mc_total_score += mc_card_value
			opp_total_score += opp_card_value
		
		# If deck isn't empty, pick up new cards
		if len(deck) != 0:
			mc_cards = np.sort(np.append(mc_cards, deck[:1]))
			deck = deck[1:]
		if len(deck) != 0:
			opp_cards = np.sort(np.append(opp_cards, deck[:1]))
			deck = deck[1:]
		
	# Determine final winner of the game and give out reward
	rew = 0
	fraction_won.write("%.3f\n" % (wins/(episode_index+1)))
	score_every_hand.write("%.2f %.2f\n" % (mc_total_score,opp_total_score))
	if mc_total_score > opp_total_score:
		rew = +1
		wins += 1
	elif mc_total_score < opp_total_score:
		rew = -1
	#rew = mc_total_score - opp_total_score # differential reward system
	
	# Accumulate these values used in computing statistics on this action value function Q^pi
	for state_index in xrange(len(monte.state_seen)):
		sta_ind = int(true_state_index[int(np.ravel_multi_index(monte.state_seen[state_index], dims=(num_cards+1,num_cards,num_cards,num_cards)))])
		act_ind = int(true_state_index[int(monte.policy_pi[sta_ind])])
		monte.update(sta_ind, act_ind, rew)
	#opp_pol = np.copy(monte.policy_pi) # update opponent (self-play)
	#if episode_index % 5000 == 0:
	#	print "...%i games played..." % episode_index
	#	np.savetxt('Q-0_ep%i.txt' % episode_index, monte.Q, fmt='%.8f')

fraction_won.close()
score_every_hand.close()

monte.print_mc(num_episodes)

end = time.clock()
print "%i games took %.2f minutes." % (num_episodes,(end-start)/60)