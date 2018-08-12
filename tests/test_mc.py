from __future__ import division

import numpy as np
import pytest

from deck_divide_dollar.q_learning import MonteCarloLearning


class TestMonteCarloLearning(object):
    def test_init(self):
        num_states = 6
        num_actions = 2
        agent = MonteCarloLearning(num_states, num_actions)

        assert agent.num_states == num_states
        assert agent.num_actions == num_actions
        assert agent.Q.shape == (num_states, num_actions)
        assert len(agent.optimal_policy) == num_states
        assert np.max(agent.optimal_policy) < num_actions
        assert agent.state_action_reward_sum.shape == (num_states, num_actions)
        assert agent.state_action_count.shape == (num_states, num_actions)
        assert len(agent.states_seen) == 0

        with pytest.raises(AssertionError):
            agent = MonteCarloLearning(0, num_actions)

        with pytest.raises(AssertionError):
            agent = MonteCarloLearning(num_states, 0)

    def test_update(self):
        num_states = 4
        num_actions = 3
        agent = MonteCarloLearning(num_states, num_actions)
        state_index = num_states - 1
        action_index = num_actions - 1

        with pytest.raises(AssertionError):
            agent.update(num_states, action_index, 0)

        with pytest.raises(AssertionError):
            agent.update(state_index, num_actions, 1)

        rewards = [1, 1, 0, -1]
        expected_count = [1, 2, 3, 4]
        expected_sum = [1, 2, 2, 1]

        for reward, exp_count, exp_sum in zip(rewards, expected_count, expected_sum):
            agent.update(state_index, action_index, reward)
            assert agent.state_action_count[state_index, action_index] == exp_count
            assert agent.state_action_reward_sum[state_index, action_index] == exp_sum
            assert agent.Q[state_index, action_index] == exp_sum / exp_count
            assert agent.optimal_policy[state_index] == action_index

        new_action_index = action_index - 1
        agent.update(state_index, new_action_index, 10)
        assert agent.optimal_policy[state_index] == new_action_index

    def test_record_state_seen(self):
        agent = MonteCarloLearning(3, 4)
        states = [[0, 1, 2], [1, 2, 1]]

        agent.record_state_seen(states[0])
        agent.record_state_seen(states[1])
        assert len(agent.states_seen) == len(states)
        assert len(agent.states_seen[-2]) == len(states[0])
        assert len(agent.states_seen[-1]) == len(states[1])
        assert all(i == j for i, j in zip(agent.states_seen[-2], states[0]))
        assert all(i == j for i, j in zip(agent.states_seen[-1], states[1]))

    def test_clear_states_seen(self):
        agent = MonteCarloLearning(3, 4)
        states = [[0], [1], [2]]

        agent.record_state_seen(states[0])
        agent.record_state_seen(states[1])
        agent.record_state_seen(states[2])
        agent.clear_states_seen()
        assert len(agent.states_seen) == 0
