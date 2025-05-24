import matplotlib.pyplot as plt
import networkx as nx
import os

def plot_training_and_partitioning(rewards, G, pred, output_dir="plots"):
    """
    Plots training reward progression and final graph partition.

    Args:
        rewards (list): List of reward values during training.
        G (networkx.Graph): The graph used during training.
        pred (np.ndarray): The predicted partition (array of 0s and 1s).
        output_dir: The output directory to store the PNG images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot training rewards
    plt.plot(rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Cut Value (Reward)")
    plt.title("Training Progress - MaxCut Reward")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_reward_plot.png"))
    plt.show()
    plt.close()

    # Plot final partition
    partition = {i: pred[i] for i in range(len(pred))}
    colors = ['red' if partition[i] == 0 else 'blue' for i in G.nodes()]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, with_labels=True, edge_color='gray')
    plt.title("Final Partitioning of Nodes")
    plt.savefig(os.path.join(output_dir, "graph_partition.png"))
    plt.show()
    plt.close()
