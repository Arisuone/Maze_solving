import matplotlib.pyplot as plt
import pygame
from env import MazeEnv, GRID_SIZE
from agent import QAgent

CELL_SIZE = 80

WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

EPISODES = 500  # 训练轮数

def train(env, agent):
    rewards_per_episode = []

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    # 绘制训练曲线并保存为图片
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Training Progress")
    plt.savefig("training.png")  # 保存为文件
    plt.show()

def run_game(agent, env):
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE))
    pygame.display.set_caption("Q-learning Maze")

    state = env.reset()
    done = False
    steps = 0
    max_steps = 200

    # 展示时关闭探索
    agent.EPSILON = 0

    while not done and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1

        screen.fill(WHITE)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if env.grid[r][c] == -1:
                    pygame.draw.rect(screen, BLACK, rect)
                elif (r,c) == env.goal:
                    pygame.draw.rect(screen, GREEN, rect)
                elif (r,c) == state:
                    pygame.draw.rect(screen, BLUE, rect)
                pygame.draw.rect(screen, (200,200,200), rect, 1)
        pygame.display.flip()
        pygame.time.delay(200)

    pygame.quit()

if __name__ == "__main__":
    env = MazeEnv()
    agent = QAgent()
    train(env, agent)   # 训练并绘制曲线（保存为 training.png）
    run_game(agent, env)  # 训练完成后直接展示