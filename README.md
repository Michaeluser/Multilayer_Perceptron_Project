# Multilayer Perceptron — From Scratch

A NumPy-only implementation of a **Multilayer Perceptron (MLP)** with manual forward and backward passes, trained on binary logic functions (XOR, OR, AND). No ML frameworks are used — backpropagation, SGD, and momentum are all implemented by hand.

## Architecture

The network consists of 3 linear layers (2→4→4→1) with alternating activation functions — Tanh, ReLU, and Sigmoid — and MSE as the loss function. Each layer handles its own weight and bias updates during backpropagation.

Supported activation functions: `Sigmoid`, `HyperTan` (tanh), `ReLU`  
Optimizer: SGD with optional momentum (`beta` parameter)

## Modes

The script has three run modes controlled by `d_appr` and `flag`:

- **Default** (`d_appr=True`) — trains on XOR, OR, and AND sequentially and logs final errors to `train_results.txt`
- **Momentum comparison** (`d_appr=False, flag=True`) — trains XOR with and without momentum, logs to `momentum_impact.txt`
- **Learning rate comparison** (`d_appr=False, flag=False`) — trains XOR at two different learning rates, logs to `learning_rate.txt`

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

```bash
python multilayer_perceptron.py
```

Configure `d_appr`, `flag`, and the `MLP(alpha, beta, batch_size)` constructor arguments at the bottom of the script before running.
