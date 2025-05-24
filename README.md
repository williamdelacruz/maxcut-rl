# Max-Cut with GNN + Reinforcement Learning

This repository explores solving the Max-Cut problem using a Graph Neural Network (GNN) trained with Reinforcement Learning (REINFORCE algorithm). The project includes:

* A GNN model to output a cut decision per node.
* A custom environment that evaluates the cut using the Max-Cut objective.
* Training using REINFORCE.
* Comparison against:

  * Random partitioning.
  * SDP upper bound (via convex optimization).
* Visualization of training progress and cut results.

## Project Structure

```
maxcut-rl/
├── data/                   # Data generation
│   ├── graps.py            # Random graph generation
├── env/                    # Max-Cut environment (gym-like)
│   ├── maxcut_env.py       # Environment generation
├── model/                  # GNN architecture
│   ├── gnn.py              # GNN creation
├── train/                  # Training logic and environment
│   ├── train.py            # Main training loop
├── config.py               # Training and model configuration
├── setup_project.py        # Project configuration
├── requirements.txt        # Dependencies
└── README.md
```

## How to Run

```bash
# Create and activate virtual environment
python -m venv maxcut-env
source maxcut-env/bin/activate  # On macOS/Linux
# maxcut-env\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Run training
python -m train.train
```

## Output

At the end of training, the script prints benchmark results:

* Final reward achieved by the GNN + RL model.
* Random partition benchmark.
* SDP upper bound.

If enabled, training rewards are plotted in `training_rewards.png`.

## Notes

* Designed for research and educational purposes.
* Suitable for experimentation and modification.
