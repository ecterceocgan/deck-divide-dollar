import numpy as np

from game import CardGame, Deck, Player
from mc import MonteCarloLearning

VALUE_OF_DOLLAR = 1.0
NUM_GAMES_TO_PLAY = 2000000

CARDS_IN_DECK = {0.25: 16, 0.50: 28, 0.75: 16}
HAND_SIZE = 5
ACTIONS = ['small_spoil', 'median', 'large_max']


class DeckBasedDivideTheDollar(object):
    """Deck-based divide-the-dollar.

    Args:
        value_of_dollar (float): the threshold used for players' scoring card value vs. nothing
        card_game (CardGame): card game specifying deck, actions, states, etc.
        players (list of Players): list of Players, first is assumed to be Monte Carlo Q-Learner
        num_games_to_play (int): number of full games of deck-based divide-the-dollar to play

    Attributes:
        value_of_dollar (float): the threshold used for players' scoring card value vs. nothing
        card_game (CardGame): card game specifying deck, actions, states, etc.
        players (list of Players): list of Players, first is assumed to be Monte Carlo Q-Learner
        num_games_to_play (int): number of full games of deck-based divide-the-dollar to play
        q_learning (MonteCarloLearning): optimal policy managed by monte carlo learning

    """

    def __init__(self, value_of_dollar, card_game, players, num_games_to_play):
        self.value_of_dollar = value_of_dollar
        self.card_game = card_game
        self.players = players
        self.num_games_to_play = num_games_to_play
        self.q_learning = MonteCarloLearning(card_game.num_states, card_game.num_actions)

    def play_games(self):
        """Play all games in order to converge to optimal policy via q-learning."""
        for episode_index in xrange(self.num_games_to_play):
            self.initialize_episode()
            self.play_rounds()
            game_result = self.scorekeeping()  # reward for monte carlo player
            self.aggregate_learning(game_result)
        self.output()

    def initialize_episode(self):
        """Initialize game by shuffling deck, and resetting players' hands and q-learning states."""
        self.deck.shuffle_deck()
        for player in self.players:
            player.reset_hand()
            player.pick_up_cards(self.deck.deal_cards(self.card_game.hand_size))
        self.q_learning.clear_states_seen()

    def play_rounds(self):
        """Play all rounds of game; players take turns going first."""
        turn_order = range(len(self.players))
        for round_index in xrange(self.card_game.num_rounds):
            sum_of_cards = 0.

            for index in turn_order:
                monte_carlo_agent = True if index == 0 else False
                sum_of_cards += self._take_turn(self.players[index], round_index, sum_of_cards,
                                                monte_carlo=monte_carlo_agent)

            if sum_of_cards <= self.value_of_dollar:
                for player in self.players:
                    player.total_score += player.last_card_played

            for player in self.players:
                player.pick_up_cards(deck.deal_cards(1))

            turn_order = turn_order[1:] + turn_order[:1]

    def _take_turn(self, player, round_index, card_showing, monte_carlo=False):
        """Select player's action given game state and play card.

        Args:
            player (Player): player currently taking a turn
            round_index (int): index of round; used for exploring starts in monte carlo methods
            card_showing (float): sum of cards played thus far in current round
            monte_carlo (boolean): whether or not the player should be used for q-learning updates

        Returns:
            card value of action played

        """
        player.set_game_state(card_showing)
        if monte_carlo:
            self.q_learning.record_state_seen(player.game_state)

        policy_index = int(self.card_game.true_state_index[int(np.ravel_multi_index(
            player.game_state, dims=(self.card_game.deck.num_unique_cards + 1,
                                     self.card_game.deck.num_unique_cards,
                                     self.card_game.deck.num_unique_cards,
                                     self.card_game.deck.num_unique_cards)))])

        if monte_carlo and (round_index <= 1):  # exploring starts
            player.next_action = np.random.choice(card_game.num_actions)
        else:
            player.next_action = player.policy[policy_index]

        return self._play_action(card_showing, player)

    def _play_action(self, card_showing, player):
        """Given player's chosen action, play associated card.

        Args:
            player (Player): player currently taking a turn
            card_showing (float): sum of cards played thus far in current round

        Returns:
            card value of action played

        """
        if card_showing == 0:  # player goes first
            if player.next_action == self.card_game.actions.index('small_spoil'):
                card_value = player.play_card(0)
            elif player.next_action == self.card_game.actions.index('large_max'):
                card_value = player.play_card(-1)
            else:
                card_value = player.play_card(self.card_game.hand_size // 2)
        else:  # opponent went first, player's turn
            if player.next_action == self.card_game.actions.index('small_spoil'):
                for c, card in enumerate(player.hand):
                    if card + card_showing > self.value_of_dollar:  # can spoil, play this card
                        card_value = player.play_card(c)
                        break
                    elif c == len(player.hand) - 1:  # can't spoil, play largest card
                        card_value = player.play_card(-1)
            elif player.next_action == self.card_game.actions.index('large_max'):
                for c, card in enumerate(np.flipud(player.hand)):
                    if card + card_showing <= self.value_of_dollar:  # can maximize, play this card
                        card_value = player.play_card(len(player.hand) - 1 - c)
                        break
                    elif len(player) - c == 0:  # can't maximize, play smallest card
                        card_value = player.play_card(0)
            else:
                card_value = player.play_card(self.card_game.hand_size // 2)

        return card_value

    def scorekeeping(self):
        """Determine winner of game (highest total score).

        Players' total scores are sorted. If there is a clear winner (no ties),
        increase that player's win total.

        Returns:.
            +1 reward if Monte Carlo learner won the game; -1 otherwise

        """
        total_scores = [player.total_score for player in self.players]
        win_order = np.argsort(total_scores)[::-1]
        if total_scores(win_order[0]) > total_scores(win_order[1]):  # must have clear winner
            self.players(win_order[0]).wins += 1
            if total_scores[0] == total_scores(win_order[0]):  # is monte carlo learner the winner?
                reward = 1
                return reward
        # TODO: draws shouldn't result in negative reward
        reward = -1
        return reward


    def aggregate_learning(self, game_result):
        """Use states seen during game and game result to update Monte Carlo q-learner.

        Args:
            game_result: result of game from Monte Carlo agent's perspective (+1 for win; -1 for loss)

        """
        pass


if __name__ == 'main':
    deck = Deck(CARDS_IN_DECK)
    card_game = CardGame(deck, ACTIONS, HAND_SIZE)
    players = [Player() for _ in xrange(num_players)]
    divide_the_dollar = DeckBasedDivideTheDollar(card_game, players, NUM_GAMES_TO_PLAY)
