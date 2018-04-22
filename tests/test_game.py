import numpy as np
import pytest

from deck_divide_dollar.game import CardGame, Deck, Player


class TestDeck(object):
    def test_init(self):
        pass

    def test_shuffle_deck(self):
        pass

    def test_deal_cards(self):
        pass


class TestCardGame(object):
    def test_init(self):
        pass


class TestPlayer(object):
    def test_init(self):
        policy = [1, 1, 1, 1]
        player = Player(policy)
        assert all(i == j for i, j in zip(policy, player.policy))
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
        assert all(i == j for i, j in zip(new_policy, player.policy))

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
            assert all(i == j for i, j in zip(hand[pos+1:], player.hand))

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
