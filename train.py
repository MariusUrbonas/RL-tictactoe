import gym
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark
from gym_tictactoe.base_agent import BaseAgent
from gym_tictactoe.human_agent import HumanAgent

from uct_agent import UCTAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def transition_function(state, action):
    board, mark = state
    env = TicTacToeEnv()
    env.board = list(board)
    env.mark = mark
    next_state, _, _, _ = env.step(action=action)
    return next_state


def ava_action_function(state):
    board, mark = state
    env = TicTacToeEnv()
    env.board = list(board)
    env.mark = mark
    return env.available_actions()


def is_terminal_function(state):
    board, mark = state
    status = check_game_status(board)
    if status == -1:
        return False
    return True


def reward_function(state):
    board, mark = state
    status = check_game_status(board)
    if status == -1:
        return None
    if status == 1:
        return {'O': 1, 'X': -1}
    elif status == 2:
        return {'O': -1, 'X': 1}
    return {'O': 0, 'X': 0}


def evaluate(episodes=1, budget=10):
    start_mark = 'O'
    env = TicTacToeEnv()
    agent = UCTAgent(transition_function, ava_action_function,
                     is_terminal_function, reward_function, budget)
    agent.mark = start_mark
    agents = [agent,
              HumanAgent('X')]
    stats = []
    for _ in range(episodes):
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            env.show_turn(True, mark)
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                env.show_result(True, mark, reward)
                stats.append(reward)
            else:
                _, mark = state
        # rotate start
        start_mark = next_mark(start_mark)
    return list(map(lambda x: x, stats))


def plot(stats):
    y_vals = np.array(list(map(np.mean, stats)))
    error = np.array(list((map(np.std, stats))))/np.sqrt(10)
    print(y_vals)
    print(error)
    plt.plot(progression, y_vals, 'k',  color='#CC4F1B')
    plt.fill_between(progression, y_vals-error, y_vals+error, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.show()


total_stats = []
start = 1
n = 500
ratio = 1.01
progression = [int(start * ratio**i) for i in range(n)]
progression = range(10, 100, 10)
print(progression)
for b in tqdm(progression):
    total_stats.append(evaluate(2, b))
plot(total_stats)
