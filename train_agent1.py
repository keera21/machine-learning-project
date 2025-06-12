import gym
import numpy as np

env = gym.make("Taxi-v3").env

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 50000

for episode in range(episodes):
    state, _ = env.reset()  # <-- unpacking here
    done = False
    truncated = False

    while not (done or truncated):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

    if (episode + 1) % 5000 == 0:
        print(f"Episode: {episode + 1}")

np.save("q_table.npy", q_table)
print("Training complete and Q-table saved!")
