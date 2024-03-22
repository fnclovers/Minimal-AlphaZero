# Minimal MuZero Implementation

Welcome to our GitHub repository featuring a minimal implementation of the MuZero algorithm. MuZero is a groundbreaking reinforcement learning algorithm developed by DeepMind, known for its exceptional performance in mastering complex games like Chess, Go, Shogi, and various Atari games. Unlike traditional approaches, MuZero can achieve this without prior knowledge of the game's rules, relying solely on learning from its interaction with the game environment. This is made possible by its ingenious integration of Monte Carlo Tree Search (MCTS) with a powerful deep neural network that learns the dynamics, policy, and value functions directly from the game's visual inputs.

## Features

This implementation is designed to be as minimal and understandable as possible while retaining the core functionality that makes MuZero unique and powerful. Here are the key features:

- **Environment Setup:** We provide a flexible framework for setting up custom game environments. This setup is easily configurable for classic board games such as Chess, Shogi, Go, and even for dynamic environments like Atari games, allowing for a wide range of applications.
  
- **MCTS with Deep Learning:** At the heart of our implementation is the integration of Monte Carlo Tree Search with a deep learning model. This combination allows the algorithm to predict the value of each game state, the best policy to follow, and the underlying dynamics of the game without explicit programming of the game's rules.
  
- **Self-Play and Training Framework:** Our framework supports self-play data generation, which is crucial for the learning process of the algorithm. It also includes a training pipeline that utilizes a shared replay buffer and employs a modular network architecture for efficient learning.

## How to Run

This implementation includes pre-configured setups for two popular games: 2048 and Connect 4. Follow these instructions to get started:

1. **Playing Pretrained Games:** To play with pretrained models, first unzip the `pretrained.tar.gz` file located in the `pretrained` directory using the command `tar -zxvf pretrained.tar.gz`. Then, to play a game, simply execute either `alpha_zero_2048.py` or `alpha_zero_connect4.py` depending on your game of choice.
   
2. **Starting Training:** To train a new model from scratch, you can start the training process by executing `alpha_zero_2048.py -t` for the game of 2048. Similar commands can be used for other games once configured.

## How to Add Your Own Game

To integrate your own game into this MuZero framework, you will need to redefine four crucial classes to create a custom environment. Here's what needs to be done:

- Define your game's specific versions of the `Game`, `Action`, `ReplayBuffer`, and `Network` classes. These classes should encapsulate the rules, actions, memory handling, and neural network model of your game, respectively.
  
- Implement the `make_network` function, which initializes your game-specific neural network.

## Code Overview

The provided Python code outlines a comprehensive MuZero framework that includes:

- **Core Classes:** Essential classes such as `MinMaxStats`, `MuZeroConfig`, `Action`, `Player`, `Node`, `ActionHistory`, `Environment`, `Game`, `ReplayBuffer`, `NetworkOutput`, `Network`, and `SharedStorage`. These classes form the backbone of the MuZero algorithm, managing game states, player actions, game dynamics, and neural network interactions.

- **Configuration Functions:** Game-specific configurations are provided through functions like `make_board_game_config`, `make_go_config`, `make_chess_config`, `make_shogi_config`, and `make_atari_config`. These functions set up the neural network and game dynamics parameters tailored to each game.

- **Main MuZero Logic:** The `muzero` function encapsulates the training and self-play logic, coordinating between the neural network training processes and self-play data generation.

- **Self-Play and MCTS:** Functions for running self-play (`run_selfplay`), playing individual games (`play_game`), and executing the Monte Carlo Tree Search algorithm (`run_mcts`).

- **Training:** The training part of the code includes functions for network training (`train_network`) and utilities for processing games and training batches.

This minimal MuZero implementation provides a solid foundation for exploring and understanding one of the most advanced reinforcement learning algorithms. It's designed to be accessible yet extendable, making it an ideal starting point for your experiments with AI in gaming.

We encourage you to dive into the code, experiment with adding your own games, and contribute to this exciting area of AI research.