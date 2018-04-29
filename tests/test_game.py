from __future__ import division

import math

import numpy as np
import pytest

from deck_divide_dollar.game import CardGame, Deck, Player


class TestDeck(object):
    def test_init(self):
        cards = {1: 5, 2: 5, 3: 10}
        deck = Deck(cards)
        assert deck.cards == cards
        assert deck.unique_cards == len(cards)
        assert deck.deck_size == sum(cards.values())
        assert deck.current_deck is not None

    def test_shuffle_deck(self):
        cards = {1: 5, 2: 5, 3: 10}
        deck = Deck(cards)
        deck.shuffle_deck()
        deck_of_cards = []
        for card, num in zip(cards.keys(), cards.values()):
            deck_of_cards += [card] * num
        assert len(deck.current_deck) == len(deck_of_cards)
        assert set(deck.current_deck).issubset(set(deck_of_cards))

    def test_deal_cards(self):
        cards = {card: 1 for card in xrange(10)}
        deck = Deck(cards)
        full_deck = deck.current_deck
        hand = deck.deal_cards(5)
        assert set(hand).issubset(set(full_deck))
        assert set(deck.current_deck).issubset(set(full_deck))
        assert len(set(hand + deck.current_deck).symmetric_difference(set(full_deck))) == 0

        with pytest.raises(AssertionError):
            deck.deal_cards(deck.deck_size + 1)


class TestCardGame(object):
    def test_init(self):
        cards = {1: 5, 2: 5, 3: 10}
        deck = Deck(cards)
        num_players = 3
        actions = ['high', 'low']
        hand_size = 3
        card_game = CardGame(deck, num_players, actions, hand_size)
        assert card_game.num_players == num_players
        assert isinstance(card_game.deck, Deck)
        assert card_game.actions == actions
        assert card_game.num_actions == len(actions)
        assert card_game.num_states == int((deck.unique_cards + 1)
                                           * (math.factorial(len(actions) + deck.unique_cards - 1))
                                           / (math.factorial(len(actions))
                                              * math.factorial(deck.unique_cards - 1)))
        assert card_game.hand_size == hand_size
        assert card_game.num_rounds == 1 + (deck.deck_size
                                            - (num_players * hand_size)) // num_players
        assert card_game.true_state_index == [0, 1, 2, -1, 3, 4, -1, -1, 5, -1, -1, -1, -1,
                                              6, 7, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1,
                                              9, 10, 11, 12, -1, 13, 14, -1, -1, 15, -1, -1,
                                              -1, -1, 16, 17, -1, -1, 18, -1, -1, -1, -1, -1,
                                              -1, -1, -1, 19, 20, 21, 22, -1, 23, 24, -1, -1,
                                              25, -1, -1, -1, -1, 26, 27, -1, -1, 28, -1, -1,
                                              -1, -1, -1, -1, -1, -1, 29, 30, 31, 32, -1, 33,
                                              34, -1, -1, 35, -1, -1, -1, -1, 36, 37, -1, -1,
                                              38, -1, -1, -1, -1, -1, -1, -1, -1, 39]


class TestPlayer(object):
    def test_init(self):
        policy = [1, 1, 1, 1]
        player = Player(policy)
        assert player.policy == policy
        assert len(player.hand) == 0
        assert player.game_state is None
        assert player.next_action is None
        assert player.last_card_played is None
        assert player.total_score == 0
        assert player.wins == 0

    def test_update_policy(self):
        policy = [1, 1, 1, 1]
        player = Player(policy)
        new_policy = [2, 2, 2, 2]
        player.update_policy(new_policy)
        assert player.policy == new_policy

    def test_pick_up_cards(self):
        policy = [1, 1, 1, 1]
        player = Player(policy)

        some_cards = [1, 3, 2]
        more_cards = [4, 5]

        with pytest.raises(AssertionError):
            player.pick_up_cards(1)

        player.pick_up_cards(some_cards)
        assert set(some_cards).issubset(set(player.hand))
        player.pick_up_cards(more_cards)
        assert set(some_cards + more_cards).issubset(set(player.hand))

        def is_sorted(cards):
            return np.all(cards[:-1] <= cards[1:])

        assert is_sorted(player.hand)

    def test_play_card(self):
        policy = [1, 1, 1, 1]
        player = Player(policy)
        hand = [1, 2, 3, 4, 5]
        player.pick_up_cards(hand)

        for pos in xrange(len(hand)):
            card_value = player.play_card(0)
            assert card_value == hand[pos]
            assert player.last_card_played == card_value
            assert isinstance(player.hand, list)
            assert player.hand == hand[pos+1:]

    def test_reset_score(self):
        policy = [1, 1, 1, 1]
        player = Player(policy)
        player.total_score += 5
        player.reset_score()
        assert player.total_score == 0

    def test_reset_wins(self):
        policy = [1, 1, 1, 1]
        player = Player(policy)
        player.wins += 5
        player.reset_wins()
        assert player.wins == 0
