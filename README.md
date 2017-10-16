## About

This work started as a final project for a machine learning course at the University of Guelph. The project compares two approaches to learning an optimal strategy for playing a deck- and turn-based version of the classic generalized Nash game called divide-the-dollar. Evolutionary computation is used to evolve players encoded as a finite-state machine, in contrast to Monte Carlo policy evaluation using reinforcement learning techniques.

After the end of the term, the research continued and was included as a chapter in my [doctoral thesis](https://atrium.lib.uoguelph.ca/xmlui/handle/10214/11483) <a name="a1"></a>[[1](#WildThesis)].

For a step-by-step look at the Monte Carlo agent's learning process in playing this game, see below.

## Deck-based divide-the-dollar

A deck-based game is a mathematical game that has been transformed into a card game in which potential moves are limited to a set of cards drawn from a fixed deck <a name="a2"></a>[[2](#AshlockSchonfeldCardGames)]. Each player must choose their strategy during each turn by selecting a card from their hand. Once chosen, the card is removed and cannot be played again.

We will examine a deck-based version of the divide-the-dollar game. Divide-the-dollar is typically a simultaneous play (generalized Nash) game in which two players each make a bid to divide a dollar. If the sum of the bids is less than one, then each player gets to keep the amount of their bid <a name="a3"></a>[[3](#BramsTaylor)]. Each player's goal is obviously to maximize the amount that they get to keep.

We will look at an iterated, two-player, turn- and deck-based version of this in which we have multiple rounds of players taking turns playing the first card.

```python
from __future__ import division
import numpy as np
import math
```

### Game parameters

First, we must specify the game itself. This includes defining the specific fractions used as cards as well as the corresponding number of these unique cards that we wish to include in a deck. In this example, we consider a deck of size 60 with three unique cards: ```¼```, ```½```, and ```¾```.

```python
cards = [0.25, 0.50, 0.75]
num_unique_cards = [16, 28, 16]
num_cards = len(cards)
deck_size = sum(num_unique_cards)
```

Following this we need to define the number of players, the number of cards each player holds in their hand, and the number of rounds which constitutes a game. In this example, players will play cards until the deck is depleted and they can no longer maintain the specified hand size. This means there will be 26 rounds during a game before a winner is determined.

```python
num_players = 2
hand_size = 5
num_rounds = 1+(deck_size-(num_players*hand_size))//num_players
```

### Actions

In order to simplify the decision-making process during each round, we define three actions available to players. These actions include playing their largest card (that maximizes score), playing their median card, and playing their smallest card (that spoils the round if playing second). 

```python
num_actions = 3
```

### Game state information

Certain information about the current state of the game is available to players during their turn. This information includes whether they are playing first or second; the values of their smallest, median, and largest cards; and the value of the cards played thus far.

|Variable|Description|
|:------:|:----------|
|```card_showing```|value of card played by opponent|
|```small```|value of player's smallest card|
|```med```|value of player's median card|
|```large```|value of player's largest card|
|```index```|0 if playing first, 1 if playing second|

The player's state at a specific moment in time is then defined as the aggregate of this game information that is available to them. As the game progresses, players move through different states as the cards in their hand change: ```game_state = [card_showing, small, med, large]```. In this case, we can infer whether the player is playing first or second by looking at the value of ```card_showing```: if it is equal to zero, they're playing first; otherwise, they're playing second. Therefore, we can reduce the dimensionality of ```game_state``` by not explicitly including ```index```.

### Monte Carlo policy evaluation

In Monte Carlo policy evaluation, we estimate the value of the expected return when starting in state *s* and taking action *a*; and we call this the *action value* of a *state–action pair*.

The expected return will depend on the reward signal and how it's defined. In this example, the reward is ```+1```, ```-1```, or ```0``` for winning, losing, or drawing the game, respectively. The goal is obviously to win games which means maximize the expected reward. Players do not receive any immediate reward after playing a hand. Due to this, players must play many rounds before finding out whether they have won or lost the game. They must learn which actions in an individual round correspond to winning the game later on.

Action values for state–action pairs will thus be in the range ```-1``` to ```+1```. If the estimated action value for a particular state–action pair is equal to ```+1```, then this means that the player eventually won the game every single time they chose that action while in that state. 

#### Defining a Monte Carlo agent

We define a Monte Carlo agent as follows:

```Q``` : a matrix that contains the Monte Carlo agent's estimated action values for every state–action pair.

```policy_pi``` : an array that prescribes the best action to choose when the Monte Carlo agent is in a particular state. The best action is simply that which has the highest action value.

```action_reward_sum``` : a matrix that records the cumulative sum of rewards seen by the Monte Carlo agent for each state–action pair over the course of the learning process.

```action_reward_count``` : a matrix that records the total number of times a particular state–action pair was observed by the Monte Carlo agent over the course of the learning process.

```state_seen``` : a list of states seen by the Monte Carlo agent during a single game.

The action value matrix, ```Q```, is calculated by dividing the cumulative reward seen (```action_reward_sum```) by the number of times the state–action pair was observed (```action_reward_count```).

```python
class MC(object):
    def __init__(self, num_states, num_actions):
        # Initialize Monte Carlo agent
        self.Q = np.zeros((num_states,num_actions))
        self.policy_pi = np.random.randint(num_actions,size=num_states)
        self.action_reward_sum = np.zeros((num_states,num_actions))
        self.action_reward_count = np.zeros((num_states,num_actions))
        self.state_seen = []
    
    def update(self, sta_ind, act_ind, reward):
        # Accumulate these values used in computing statistics on this action value function Q^pi
        self.action_reward_count[sta_ind, act_ind] += 1
        self.action_reward_sum[sta_ind, act_ind] += reward
        self.Q[sta_ind, act_ind] = self.action_reward_sum[sta_ind, act_ind]/self.action_reward_count[sta_ind, act_ind]
        # Policy is greedy choice
        self.policy_pi[sta_ind] = np.argmax(self.Q[sta_ind])
    
    def record_state_seen(self, game_state):
        # Keep a record of the states seen so that action values can be updated at the end of the game
        self.state_seen.append(np.array(game_state))
    
    def clear_states_seen(self):
        # Clear the states seen (at the end of the game)
        self.state_seen = []
```

## References

<a name="WildThesis"></a> [[1](#a1)] E. Wild. [A study of heuristic approaches for solving generalized Nash equilibrium problems and related games](https://atrium.lib.uoguelph.ca/xmlui/handle/10214/11483). PhD thesis, *University of Guelph*, 2017.

<a name="AshlockSchonfeldCardGames"></a> [[2](#a2)] D. Ashlock and J. Schonfeld. [Tools for deriving card games from mathematical games](http://eldar.mathstat.uoguelph.ca/dashlock/eprints/GTRY16.pdf). *Game & Puzzle Design*, 1(2):1–3, 2015.

<a name="BramsTaylor"></a> [[3](#a3)] S. J. Brams and A. D. Taylor. [Divide the dollar: three solutions and extensions](https://doi.org/10.1007/BF01079266). *Theory and Decision*, 37:211–231, 1994.