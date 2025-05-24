import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from model.gnn import GNNMaxCut
from env.maxcut_env import MaxCutEnv
from data.graphs import generate_random_graph
from config import config
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cvxpy as cp

rewards = []

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


# def train():
#     G = generate_random_graph(config["graph"]["n_nodes"], config["graph"]["edge_prob"])
#     data = from_networkx(G)
#     data.x = torch.eye(config["graph"]["n_nodes"])  # Entrada: identidad

#     model = GNNMaxCut(in_dim=data.x.size(1), hidden_dim=config["model"]["hidden_dim"])
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

#     for epoch in range(config["training"]["epochs"]):
#         model.train()
#         optimizer.zero_grad()

#         out = model(data.x, data.edge_index)  # Log-softmax [n, 2]
#         probs = torch.exp(out)  # Convertimos a probabilidad

#         # Sampleamos acciones (partición 0 o 1) para cada nodo
#         distribs = torch.distributions.Categorical(probs)
#         actions = distribs.sample()
#         log_probs = distribs.log_prob(actions)

#         # Evaluamos la acción (partición) con el entorno
#         pred = actions.numpy()
#         env = MaxCutEnv(G)
#         _, reward, _, _ = env.step(pred)

#         # REINFORCE: pérdida = -log_prob * reward
#         loss = -log_probs.mean() * reward
#         loss.backward()
#         optimizer.step()

#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}, Reward: {reward:.4f}")

#         # save the reward
#         rewards.append(reward)

#     plt.plot(rewards)
#     plt.xlabel("Epoch")
#     plt.ylabel("Cut Value (Reward)")
#     plt.title("Training Progress - MaxCut Reward")
#     plt.grid(True)
#     plt.savefig("training_reward_plot.png")
#     plt.show()

#     partition = {i: pred[i] for i in range(len(pred))}
#     colors = ['red' if partition[i] == 0 else 'blue' for i in G.nodes()]
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, node_color=colors, with_labels=True, edge_color='gray')
#     plt.title("Final Partitioning of Nodes")
#     plt.savefig("graph_partition.png")
#     plt.show()





def train():
    import torch
    import torch.nn.functional as F
    from torch_geometric.utils import from_networkx
    import numpy as np

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


    plt.plot(rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Cut Value (Reward)")
    plt.title("Training Progress - MaxCut Reward")
    plt.grid(True)
    plt.savefig("training_reward_plot.png")
    plt.show()
    plt.close()  # Cierra el gráfico sin mostrarlo

    partition = {i: pred[i] for i in range(len(pred))}
    colors = ['red' if partition[i] == 0 else 'blue' for i in G.nodes()]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, with_labels=True, edge_color='gray')
    plt.title("Final Partitioning of Nodes")
    plt.savefig("graph_partition.png")
    plt.show()
    plt.close()  # Cierra el gráfico sin mostrarlo



if __name__ == "__main__":
    train()
