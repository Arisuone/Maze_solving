import numpy as np
import random
from collections import deque

GRID_SIZE = 6  # 迷宫大小（6x6）

def is_reachable(grid, start, goal):
    # 使用 BFS 检查起点和终点是否连通
    visited = set()
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] != -1 and (nr,nc) not in visited:
                visited.add((nr,nc))
                queue.append((nr,nc))
    return False

class MazeEnv:
    def __init__(self):
        # 初始化迷宫环境，保证起点和终点连通
        while True:
            self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
            for _ in range(GRID_SIZE):
                r, c = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                self.grid[r][c] = -1

            self.start = (0,0)
            self.goal = (GRID_SIZE-1, GRID_SIZE-1)
            self.grid[self.start] = 0
            self.grid[self.goal] = 0

            if is_reachable(self.grid, self.start, self.goal):
                break

        self.state = self.start

    def reset(self):
        # 重置环境，返回起点状态
        self.state = self.start
        return self.state

    def step(self, action):
        # 执行动作：0=上, 1=下, 2=左, 3=右
        r, c = self.state
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1

        if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE or self.grid[r][c] == -1:
            return self.state, -5, False

        self.state = (r, c)

        if self.state == self.goal:
            return self.state, 10, True

        return self.state, -1, False