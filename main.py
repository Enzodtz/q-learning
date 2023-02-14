import gymnasium as gym
import numpy as np
import random

from utils import dict_argmax

env_name = "FrozenLake-v1"
epsilon = 0.1
alpha = 0.1
gamma = 0.9
n = 2000

env = gym.make(env_name, is_slippery=False)
max_r = 0

q = {}

for s in range(env.observation_space.n):
    q[s] = {}
    for a in range(env.action_space.n):
        q[s][a] = 0

for i in range(n):
    s, _ = env.reset()
    done = False
    rwrds = 0
    t = 0
    while not done:
        if random.random() < epsilon:
            a = env.action_space.sample()
        else:
            a = dict_argmax(q[s])

        s_prime, r, done, _, _ = env.step(a)

        q[s][a] = q[s][a] + alpha * (
            r + gamma * q[s_prime][dict_argmax(q[s_prime])] - q[s][a]
        )

        if r == 1:
            print("reward 1")

        s = s_prime
        rwrds += r
        t += 1

    max_r = max(max_r, rwrds)
    print(f"Episode [{i}], Duration [{t}] Reward [{rwrds}], Max [{max_r}]")

# Visualization
v = {}
for s in q:
    v[s] = max(q[s], key=q[s].get)
v = list(v.items())
for i in range(4):
    print([v for k, v in v[4 * i : 4 * (1 + i)]])

human_env = gym.make(env_name, render_mode="human", is_slippery=False)

while True:
    s, _ = human_env.reset()
    done = False

    while not done:
        a = dict_argmax(q[s])
        s, r, done, _, _ = human_env.step(a)
