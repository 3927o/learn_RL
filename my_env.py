import time

import pandas as pd
import numpy as np


class Map:
    def __init__(self, env):
        self.map = env
        self.position = (0, 0)
        self.width = len(self.map[0])
        self.height = len(self.map)

    def update_position(self, x, y):
        self.position = (x, y)

    def step(self, action):
        x, y = self.position
        x_updated, y_updated = x, y
        if action == 0:
            x_updated = x-1
        elif action == 1:
            x_updated = x+1
        elif action == 2:
            y_updated = y-1
        elif action == 3:
            y_updated = y+1
        else:
            raise Exception("Wrong Action")

        if not 0 <= x_updated < self.height or not 0 <= y_updated < self.width or self.map[x_updated][y_updated] == "*":
            x_updated, y_updated = x, y

        print(f"action is {action}, x now is {x_updated}, y now is {y_updated}")
        self.update_position(x_updated, y_updated)
        self.show()
        return self.map[self.position[0]][self.position[1]] == "T"

    def show(self):
        env = self.map
        temp = env[self.position[0]][self.position[1]]
        env[self.position[0]][self.position[1]] = "O"
        out = "\n".join(["".join(line) for line in env])
        print(out)
        env[self.position[0]][self.position[1]] = temp
        time.sleep(0.1)


epoch_num = 100
env = [
            ['-', '-', '-', '-', '-', ],
            ['-', '*', '-', '*', '-', ],
            ['-', '*', '-', 'T', '-', ],
            ['-', '*', '-', '-', '-', ],
            ['-', '-', '-', '-', '-', ],
        ]
GAMMA = 0
ALPHA = 0.1


class RL:
    def __init__(self):
        self.env = Map(env)
        self.actions = range(4)
        self.q_table = pd.DataFrame(np.zeros((self.env.width*self.env.height, len(self.actions))),
                                    columns=self.actions)  # columns 对应的是行为名称
        self.epsilon = 0.9

    def get_state(self):
        return self.env.position[0] * self.env.width + self.env.position[1]

    def choose_action(self, state):
        state_actions = self.q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
        if (np.random.uniform() > self.epsilon) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
            action_name = np.random.choice(self.actions)
        else:
            action_name = self.actions[state_actions.argmax()]  # 贪婪模式
        return action_name

    def train(self):
        for epoch in range(epoch_num):
            print(f"executing epoch {epoch}")
            self.env.update_position(0, 0)
            is_terminated = False
            step = 0
            while not is_terminated:
                state = self.get_state()
                action = self.choose_action(state)
                is_terminated = self.env.step(action)
                state_ = self.get_state()
                R = 100 / step if is_terminated else 0
                step += 1

                q_predict = self.q_table.loc[state, action]
                q_target = R + GAMMA * self.q_table.loc[state_, :].max()

                self.q_table.loc[state, action] += ALPHA * (q_target - q_predict)

            print(f"epoch {epoch} finished, step cnt is {step}")

        print(self.q_table)


if __name__ == '__main__':
    rl = RL()
    rl.train()
