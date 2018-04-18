import numpy as np

from deck_divide_dollar.mc import MonteCarloLearning


def test_mc_init():
    num_states = 6
    num_actions = 2
    agent = MonteCarloLearning(num_states, num_actions)

    assert agent.Q.shape == (num_states, num_actions)
    assert len(agent.optimal_policy) == num_states
    assert np.max(agent.optimal_policy) < num_actions
    assert agent.state_action_reward_sum.shape == (num_states, num_actions)
    assert agent.state_action_count.shape == (num_states, num_actions)
    assert len(agent.states_seen) == 0


def test_record_state_seen():
    agent = MonteCarloLearning(3, 4)
    state = [0, 1, 2]

    agent.record_state_seen(state)
    assert len(agent.states_seen[-1]) == len(state)
    assert all(i == j for i, j in zip(agent.states_seen[-1], state))
