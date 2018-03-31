from __future__ import division

import math

import numpy as np


class Deck(object):
    """Deck.

    Args:
        cards (dict): {card_value: unique_cards}

    Attributes:
        cards (dict): {card_value: num_cards}
        unique_cards (int): number of unique card values
        deck_size (int): total number of cards in the deck
        current_deck (list): full list of cards remaining in deck

    """

    def __init__(self, cards):
        """Initialize deck of cards."""
        self.cards = cards
        self.unique_cards = len(self.cards)
        self.deck_size = sum(self.cards.values())
        self.current_deck = None
        self.shuffle_deck()

    def shuffle_deck(self):
        """Initialize shuffled deck of (all) cards.

        Returns:
            (array): shuffled deck of cards

        """
        d = []
        for card, num in zip(self.cards.keys(), self.cards.values()):
            d += [card] * num
        self.current_deck = np.random.shuffle(np.array(d))

    def deal_cards(self, num_cards_to_deal):
        assert len(self.current_deck) >= num_cards_to_deal, \
            'Not enough cards left in deck to deal those cards.'
        dealt_cards = self.current_deck[:num_cards_to_deal]
        self.current_deck = self.current_deck[num_cards_to_deal:]
        return dealt_cards


class CardGame(object):
    """Card game.

    Args:
        num_players (int): number of players playing the game
        deck (Deck): deck of cards
        actions (list): list of action names
        hand_size (int): number of cards a player holds in their hand

    Attributes:
        num_players (int): number of players playing the game
        deck (Deck): deck of cards
        actions (list): list of action names
        num_actions (int): number of unique actions
        num_states (int): number of possible game states
        hand_size (int): number of cards a player holds in their hand
        num_rounds (int): number of rounds that are played in one game

    """

    def __init__(self, deck, num_players, actions, hand_size):
        """Initialize card game.

        TODO: factorial 3??
        """
        self.num_players = num_players
        self.deck = deck
        self.actions = actions
        self.num_actions = len(actions)
        self.num_states = int((self.deck.unique_cards + 1)
                              * (math.factorial(3 + self.deck.unique_cards - 1))
                              / (math.factorial(3) * math.factorial(self.deck.unique_cards - 1)))
        self.hand_size = hand_size
        self.num_rounds = 1 + (self.deck.deck_size
                               - (self.num_players * self.hand_size)) // self.num_players
        self.true_state_index = self._true_state_index()

    def _true_state_index(self):
        """Return the true index in list of unique states for each permutation.

        For a potential game state permutation [card_showing, smallest, median, largest],
        if smallest <= median <= largest does not hold, the permutation is an invalid game state
        and the true state index should be a -1. For all valid permutations, the true state index
        should be sequentially increasing.

        Returns:
            (list): true state index of valid permutations

        """
        states = []
        unique_cards = self.deck.unique_cards
        for card_showing in xrange(unique_cards + 1):
            for smallest in xrange(unique_cards):
                for median in xrange(unique_cards):
                    for largest in xrange(unique_cards):
                        states.append(np.array([card_showing, smallest, median, largest]))

        true_state_index = []
        true_index_counter = 0
        for state in states:
            if np.all(state[1:-1] <= state[2:]):
                true_state_index.append(true_index_counter)
                true_index_counter += 1
            else:
                true_state_index.append(-1)

        return true_state_index


class Player(object):
    def __init__(self, policy):
        self.policy = policy
        self.hand = []
        self.game_state = None
        self.next_action = None
        self.last_card_played = None

    def update_policy(self, policy):
        self.policy = policy

    def pick_up_cards(self, cards):
        self.hand = np.sort(self.hand.append(cards))

    def play_card(self, card_position_in_hand):
        card_value = self.hand[card_position_in_hand]
        self.hand = np.delete(self.hand, 0)
        self.last_card_played = card_value
        return card_value

    def set_game_state(self, card_showing):
        median_card_index = len(self.hand) // 2
        self.game_state = [card_showing, self.hand[0], self.hand[median_card_index], self.hand[-1]]
