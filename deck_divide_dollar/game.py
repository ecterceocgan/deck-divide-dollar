import math
import random


class Deck(object):
    """Deck of cards.

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
        self.current_deck = self.shuffle_deck()

    def __repr__(self):
        return 'Deck(cards=%s)' % repr(self.cards)

    def shuffle_deck(self):
        """Return shuffled deck of all cards."""
        deck = [card for card, num in zip(self.cards.keys(), self.cards.values())
                     for n in xrange(num)]
        random.shuffle(deck)
        return deck

    def deal_cards(self, num_cards_to_deal):
        """Deal N cards from top of deck."""
        assert len(self.current_deck) >= num_cards_to_deal, \
            'Not enough cards left in deck to deal those cards.'
        dealt_cards = self.current_deck[:num_cards_to_deal]
        self.current_deck = self.current_deck[num_cards_to_deal:]
        return dealt_cards


class Player(object):
    """Player of card game.

    Args:
        policy (list): an initial policy

    Attributes:
        policy (list): the player's optimal policy for choosing action when in state
        hand (list): the player's cards in hand
        game_state (array): the current state of the game
        next_action: the player's chosen action to play next turn
        last_card_played (float): the value of the last card played
        total_score (float): the player's total score
        wins (int): the player's total number of wins

    """

    hand_size = 7

    def __init__(self, policy=None):
        """Initialize player."""
        self.policy = policy
        self.hand = []
        self.game_state = None
        self.next_action = None
        self.last_card_played = None
        self.total_score = 0
        self.wins = 0

    def __repr__(self):
        return 'Player(policy=%s)' % repr(self.policy)

    def pick_up_cards(self, cards):
        """Add cards (list) to player's hand."""
        assert isinstance(cards, (list))
        self.hand = sorted(self.hand + cards)

    def play_card(self, card_position_in_hand):
        """Play specific card by position in hand."""
        card_value = self.hand[card_position_in_hand]
        del self.hand[card_position_in_hand]
        self.last_card_played = card_value
        return card_value

    def set_game_state(self, card_showing):
        """TODO: Remove this method; not appropriate for player class."""
        median_card_index = len(self.hand) // 2
        self.game_state = [card_showing, self.hand[0], self.hand[median_card_index], self.hand[-1]]

    def reset_hand(self):
        """Remove all cards from a player's hand."""
        self.hand = []

    def reset_score(self):
        """Reset player's score to zero."""
        self.total_score = 0

    def reset_wins(self):
        """Reset player's win count to zero."""
        self.wins = 0
