import torch
import torch.nn.functional as F
from   torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cvxpy as cp

# Local scripts

from model.gnn import GNNMaxCut
from env.maxcut_env import MaxCutEnv
from data.graphs import generate_random_graph
from config import config
from utils.visualization import plot_training_and_partitioning


def compute_random_cut_value(graph):
    n = graph.number_of_nodes()
    partition = np.random.randint(0, 2, size=n)
    cut_value = 0
    for u, v, data in graph.edges(data=True):
        if partition[u] != partition[v]:
            cut_value += data.get("weight", 1.0)
    return cut_value

def compute_sdp_upper_bound(graph):
    n = graph.number_of_nodes()
    W = np.zeros((n, n))
    for u, v, data in graph.edges(data=True):
        W[u, v] = W[v, u] = data.get("weight", 1.0)

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == 1]
    objective = cp.Maximize(0.25 * cp.sum(cp.multiply(W, (1 - X))))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    return prob.value



def train():
    rewards = []

    # Genera un grafo aleatorio
    G = generate_random_graph(config["graph"]["n_nodes"], config["graph"]["edge_prob"])
    data = from_networkx(G)
    data.x = torch.eye(data.num_nodes)  # Input como identidad

    # Inicializa modelo y optimizador
    model = GNNMaxCut(in_dim=data.x.size(1), hidden_dim=config["model"]["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    best_reward = -float("inf")
    best_pred = None
    rewards = []  # Almacena recompensas por época

    for epoch in range(config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)  # Salida softmax por nodo
        probs = torch.exp(out)

        # Distribución categórica para muestreo
        distribs = torch.distributions.Categorical(probs)
        actions = distribs.sample()  # Asignación de partición por nodo
        log_probs = distribs.log_prob(actions)

        # Evaluar acción en el entorno
        pred = actions.numpy()
        env = MaxCutEnv(G)
        _, reward, _, _ = env.step(pred)

        # Guardar mejor recompensa
        if reward > best_reward:
            best_reward = reward
            best_pred = pred

        # Algoritmo REINFORCE
        loss = -log_probs.mean() * reward
        loss.backward()
        optimizer.step()

        rewards.append(reward)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Reward: {reward:.4f}")

    # Evaluación Benchmark
    random_reward = compute_random_cut_value(G)
    sdp_bound = compute_sdp_upper_bound(G)

    print("\n=== Benchmark Final ===")
    print(f"GNN + RL Reward: {best_reward:.4f}")
    print(f"Random Reward:   {random_reward:.4f}")
    print(f"SDP Upper Bound: {sdp_bound:.4f}")


    # Create plots
    plot_training_and_partitioning(rewards, G, best_pred)



if __name__ == "__main__":
    train()
