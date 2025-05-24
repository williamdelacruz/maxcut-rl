import gym
from gym import spaces
import numpy as np

class MaxCutEnv(gym.Env):
    def __init__(self, graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.action_space = spaces.Discrete(2 ** self.n)  # Simplificado
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n, self.n), dtype=np.float32)
        self.state = np.zeros(self.n, dtype=int)

    def reset(self):
        self.state = np.random.randint(0, 2, size=self.n)
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        # Acción es un vector binario indicando asignación de particiones
        self.state = action
        reward = self._compute_cut_value()
        done = True  # Un solo paso
        return self._get_obs(), reward, done, {}

    def _compute_cut_value(self):
        cut_value = 0
        for u, v, data in self.graph.edges(data=True):
            if self.state[u] != self.state[v]:
                cut_value += data.get("weight", 1.0)
        return cut_value
