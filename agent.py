import numpy as np
import random
from env import GRID_SIZE

# Q-learning 参数
ALPHA = 0.1   # 学习率
GAMMA = 0.9   # 折扣因子
EPSILON = 0.2 # 探索率

class QAgent:
    def __init__(self):
        # 初始化 Q-table
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.EPSILON = EPSILON

    def choose_action(self, state):
        # epsilon-greedy 策略
        if random.uniform(0,1) < self.EPSILON:
            return random.randint(0,3)
        r, c = state
        return np.argmax(self.q_table[r][c])

    def learn(self, state, action, reward, next_state):
        # Q-learning 更新公式
        r, c = state
        nr, nc = next_state
        predict = self.q_table[r][c][action]
        target = reward + GAMMA * np.max(self.q_table[nr][nc])
        self.q_table[r][c][action] += ALPHA * (target - predict)