import copy
import time

import torch

from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from Model import Model, QTrainer
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from TetrisBattle.tetris import get_infos
from sklearn.preprocessing import StandardScaler
from TetrisBattle.tetris import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


class Agent:
    def __init__(self, mode="single"):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.mode = mode
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Model().to('cuda')
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, env):
        game_interface = env.game_interface
        game = game_interface.tetris_list[game_interface.now_player]["tetris"]
        return np.array(get_infos(game.get_board()))
        # if self.mode == "single":
        #     return np.squeeze(env.game_interface.get_obs(), axis=-1)[:, :17].flatten()
        # else:
        #     return np.squeeze(env.game_interface.get_obs(), axis=-1).flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            BATCH = random.sample(self.memory, BATCH_SIZE)
        else:
            BATCH = self.memory

        state, action, reward, next_state, done = zip(*BATCH)
        return self.trainer.train_step(state, action, reward, next_state, done, long_term=True)

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_action = [0] * 8

        epsilon = 1e-3 + (max(80 - self.n_games, 0) * (
                1 - 1e-3) / 2000)
        u = random.random()
        random_action = u < epsilon

        if random_action:
            action_idx = random.randint(0, 7)
            final_action[action_idx] = 1
        else:
            stat0 = torch.tensor(state).unsqueeze(0).to('cuda').type(torch.float)
            prediction = self.model(stat0)
            action_idx = torch.argmax(prediction, dim=-1).cpu().item()
            final_action[action_idx] = 1
        return final_action

    def get_next_state(self, env: TetrisSingleEnv):
        """
        return list of all posiblle stattes .
        """
        game_interface = env.game_interface
        game = game_interface.tetris_list[game_interface.now_player]["tetris"]
        # print(game.get_board())  # Lấy Ma Trận Game Hiện Tại [ 10 * 20]
        # block_class = game.block  # Trả về class Piece
        # block = block_class.now_block()  # Lấy piece hiện tại
        # block_class.rotate(_dir=1)  # Xoay block hiện tại 1 vòng theo chiều kim đồng hồ
        # # Di chuyển sang trái sang phải
        # print(game.px, game.py)  # Lấy vị trí hiện tại của piece
        # # collideLeft,collideRight check xem cái piece của mình đã chạm vào tường trái hay chưa
        # gap_y = hardDrop()  # khoảng cách từ vị trí hiện tại đến vị trí cuối cùng có thể đặt theo chiều y

        states_list = []

        for _dir in range(4):
            check = 0
            for i in range(12):
                game_copy = copy.deepcopy(game)
                grid = game_copy.get_board()
                # Xoay và lấy trạng thái hiện tại của block
                old_block = game_copy.block.now_block()
                game_copy.block.rotate(_dir=_dir)  # Xoay block
                block = game_copy.block.now_block()
                if np.array_equal(np.array(old_block), np.array(block)):
                    check = 1
                for x in range(1, i):
                    if not collideLeft(grid, game_copy.block, game_copy.px, game_copy.py):
                        game_copy.px -= 1
                    else:
                        break

                add_y = hardDrop(grid, game_copy.block, game_copy.px, game_copy.py)
                excess = len(grid[0]) - 20
                for x in range(4):
                    for y in range(4):
                        if block[x][y]:
                            grid[x + game_copy.px][y + game_copy.py + add_y - excess] = 1
                states_list.append(grid)

            for i in range(12):
                game_copy = copy.deepcopy(game)
                grid = game_copy.get_board()
                # Xoay và lấy trạng thái hiện tại của block
                game_copy.block.rotate(_dir=_dir)  # Xoay block
                block = game_copy.block.now_block()
                for x in range(1, i):
                    if not collideRight(grid, game_copy.block, game_copy.px, game_copy.py):
                        game_copy.px += 1
                    else:
                        break

                add_y = hardDrop(grid, game_copy.block, game_copy.px, game_copy.py)
                excess = len(grid[0]) - 20
                for x in range(4):
                    for y in range(4):
                        if block[x][y]:
                            grid[x + game_copy.px][y + game_copy.py + add_y - excess] = 1
                states_list.append(grid)
            if check == 1:
                break
        return states_list


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")

    agent = Agent()

    # agent.n_games = 82
    # agent.model.load_state_dict(torch.load('model.pth'))
    # agent.n_games = 118
    # agent.model.load_state_dict(torch.load('model_.pth'))
    while True:
        # Get Current State
        state_old = agent.get_state(env)

        # Get Action
        action = np.array(agent.get_action(state_old))
        action_idx = np.where(action != 0)[0][0]
        action = env.random_action()
        # Perform Action
        ob, reward, done, infos = env.step(action)
        print(len(agent.get_next_state(env)))
        # Get New State
        state_new = agent.get_state(env)

        loss_short = agent.train_short_memory(state_old, action, reward, state_new, done)

        agent.remember(state_old, action, reward, state_new, done)
        if done:
            ob = env.reset()
            agent.n_games += 1

            loss_long = agent.train_long_memory()

            if infos['scores'] >= record:
                record = infos['scores']
                agent.model.save(file_name='model_.pth')
            print('Game', agent.n_games, 'Score', infos['scores'], 'Record', record)
            print('Loss', loss_long)
            plot_scores.append(infos['scores'])
            total_score += infos['scores']
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)


if __name__ == '__main__':
    train()
