import gym
import numpy as np
import pygame
import time
import os

# Constants
CELL_SIZE = 100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TAXI_COLOR = (255, 255, 0)
PICKUP_COLOR = (0, 255, 0)
DROPOFF_COLOR = (255, 0, 0)

# Pygame Init
pygame.init()
screen = pygame.display.set_mode((5 * CELL_SIZE, 5 * CELL_SIZE))
pygame.display.set_caption("Taxi-v3 GUI")

# Environment Setup
env = gym.make("Taxi-v3", render_mode="rgb_array")
env = env.unwrapped  # Needed for decode()
q_table_file = "q_table.npy"

# ---- TRAIN Q-TABLE IF NOT EXISTS ----
if not os.path.exists(q_table_file):
    print("Training Q-table...")

    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    decay = 0.9995

    for ep in range(25000):
        state, _ = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _, _ = env.step(action)

            # Q-learning update
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            state = next_state

        epsilon = max(epsilon * decay, epsilon_min)

        if ep % 2000 == 0:
            print(f"Episode {ep} | epsilon={epsilon:.3f}")

    np.save(q_table_file, q_table)
    print("✅ Training complete. Saved to q_table.npy")
else:
    q_table = np.load(q_table_file)
    print("📂 Loaded existing q_table.npy")

# ---- DRAW FUNCTION ----
def draw_env(state):
    screen.fill(WHITE)
    taxi_row, taxi_col, pass_idx, dest_idx = env.decode(state)

    # Draw grid
    for r in range(5):
        for c in range(5):
            pygame.draw.rect(screen, BLACK, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE), 2)

    locations = [(0, 0), (0, 4), (4, 0), (4, 3)]

    # Passenger location (if not in taxi)
    if pass_idx != 4:
        px, py = locations[pass_idx]
        pygame.draw.circle(screen, PICKUP_COLOR, (py * CELL_SIZE + 50, px * CELL_SIZE + 50), 10)

    # Destination
    dx, dy = locations[dest_idx]
    pygame.draw.circle(screen, DROPOFF_COLOR, (dy * CELL_SIZE + 50, dx * CELL_SIZE + 50), 10)

    # Taxi
    pygame.draw.rect(screen, TAXI_COLOR, (taxi_col * CELL_SIZE + 25, taxi_row * CELL_SIZE + 25, 50, 50))
    pygame.display.flip()

# ---- RUN EPISODE FUNCTION ----
def run_episode():
    state, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        pygame.event.pump()  # keeps window responsive
        draw_env(state)
        time.sleep(0.6)

        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, _ = env.step(action)

        print(f"Step {steps} | Action: {action} | Reward: {reward}")
        state = next_state
        steps += 1

        if done or truncated:
            draw_env(state)
            print("✅ Trip completed.")
            time.sleep(3)  # Show final frame
            break

# ---- MAIN LOOP ----
def main():
    run_episode()
    pygame.quit()

if __name__ == "__main__":
    main()
