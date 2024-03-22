#!/usr/bin/env python
import alpha_zero
import collections
import typing
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
import pickle as pkl
import os
import torch.multiprocessing as mp
import time
import resource
import sys

BOARD_SIZE = 4


class Action2048(alpha_zero.Action):

    def __init__(self, index: int):
        super().__init__(index)

    def __str__(self) -> str:
        # "up", "down", "left", "right"
        return ["up", "down", "left", "right"][self.index]


# redefine the game
class Game2048(alpha_zero.Game):
    def __init__(self, action_space_size: int, discount: float):
        super().__init__(action_space_size, discount)
        self.size = BOARD_SIZE
        board = np.zeros((self.size, self.size), dtype=np.int32)
        self.boards = [board]
        self.add_random_tile(board)
        self.add_random_tile(board)

    def terminal(self) -> bool:
        board = self.boards[-1]
        if np.any(board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if board[i, j] == board[i, j + 1] or board[j, i] == board[j + 1, i]:
                    return False
        return True

    def legal_actions(self) -> List[alpha_zero.Action]:
        board = self.boards[-1]
        moves = []
        if self.can_merge_up(board):
            moves.append(alpha_zero.Action(0))
        if self.can_merge_left(board):
            moves.append(alpha_zero.Action(2))

        # if len(moves) != 0:
        #     return moves

        if self.can_merge_up(np.rot90(board, 2)):
            moves.append(alpha_zero.Action(1))
        if self.can_merge_left(np.fliplr(board)):
            moves.append(alpha_zero.Action(3))

        return moves

    def apply(self, action: alpha_zero.Action):
        board = self.boards[-1]
        if action.index == 0:
            reward, new_board = self.merge_up(board)
        elif action.index == 1:
            reward, new_board = self.merge_up(np.rot90(board, 2))
            new_board = np.rot90(new_board, 2)
        elif action.index == 2:
            reward, new_board = self.merge_left(board)
        elif action.index == 3:
            reward, new_board = self.merge_left(np.fliplr(board))
            new_board = np.fliplr(new_board)
        self.add_random_tile(new_board)

        self.boards.append(new_board)
        self.rewards.append(reward)
        self.history.append(action)

    def make_image(self, state_index: int):
        return self.boards[state_index]

    def add_random_tile(self, board):
        empty_cells = np.argwhere(board == 0)
        if len(empty_cells) > 0:
            cell = empty_cells[np.random.choice(len(empty_cells))]
            board[cell[0], cell[1]] = 2 if np.random.random() < 0.9 else 4

    def can_merge_left(self, board):
        for i in range(self.size):
            row = board[i, :]
            for j in range(self.size - 1):
                if row[j] == 0 and row[j + 1] > 0:
                    return True
                if (row[j] == row[j + 1]) and row[j] > 0:
                    return True
        return False

    def can_merge_up(self, board):
        return self.can_merge_left(board.T)

    def merge_left(self, board):
        reward = 1
        new_board = np.zeros_like(board)
        for i in range(self.size):
            j = 0
            k = 0
            last_value = 0
            while j < self.size:
                if board[i, j] == 0:
                    j += 1
                else:
                    if last_value == board[i, j]:
                        reward += last_value
                        new_board[i, k - 1] = 2 * last_value
                        last_value = 0
                        j += 1
                    else:
                        new_board[i, k] = board[i, j]
                        last_value = board[i, j]
                        j += 1
                        k += 1

        return reward, new_board

    def merge_up(self, board):
        reward, new_board = self.merge_left(board.T)
        return reward, new_board.T

    def get_score(self, state_index: int):
        return np.sum(self.boards[state_index])

    def print_game(self, state_index: int) -> str:
        if state_index == -1:
            state_index = len(self.boards) - 1
        board = self.boards[state_index]
        if state_index == 0:
            print("############### Initial State ###############")
        else:
            print(
                f"########### State {state_index} | Action {self.history[state_index - 1]} | Value {self.root_values[state_index]:.2f} ###########"
            )
        for i in range(self.size):
            print(
                f"{board[i, 0]:4d} {board[i, 1]:4d} {board[i, 2]:4d} {board[i, 3]:4d}"
            )
        print(f"##############################################")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.fc = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 2)
            + [nn.Linear(hidden_dim, output_dim)]
        )

    def forward(self, x):
        for layer in self.fc[:-1]:
            x = F.relu(layer(x))
        x = self.fc[-1](x)
        return x


class Network2048(nn.Module, alpha_zero.Network):
    def __init__(self, input_dim, num_actions, training, device="cpu"):
        super().__init__()

        # parameters
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = device
        self.n_training_steps = 0
        self.hidden_dim = 64
        self.representation_layers = 4
        self.dynamic_layers = 4
        self.prediction_layers = 4
        self.reduced_channels = 16
        self.fc_hidden_dim = 64
        self.fc_layers = 3
        self.support_size = input_dim * input_dim + 5
        self.training = training

        # for the representation function
        max_depth = input_dim * input_dim
        power_of_two = 2 ** torch.arange(1, max_depth + 1)
        self.power_of_two = torch.cat((torch.tensor([0]), power_of_two))
        self.power_of_two = self.power_of_two.view(1, -1, 1, 1).to(device)
        self.representation_conv = nn.Conv2d(
            max_depth + 1, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.representation_res_blocks = nn.Sequential(
            *[ResidualBlock(self.hidden_dim) for _ in range(self.representation_layers)]
        )

        # for the dynamics function
        self.dynamic_conv = nn.Conv2d(
            (max_depth + 1) + self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.dynamic_res_blocks = nn.Sequential(
            *[ResidualBlock(self.hidden_dim) for _ in range(self.dynamic_layers)]
        )
        self.reward_conv = nn.Conv2d(
            self.hidden_dim, self.reduced_channels, kernel_size=1
        )
        self.reward_fc = MLP(
            self.reduced_channels * input_dim * input_dim,
            self.fc_hidden_dim,
            self.support_size,
            self.fc_layers,
        )

        # for the prediction function
        self.prediction_res_blocks = nn.Sequential(
            *[ResidualBlock(self.hidden_dim) for _ in range(self.prediction_layers)]
        )
        self.value_conv = nn.Conv2d(
            self.hidden_dim, self.reduced_channels, kernel_size=1
        )
        self.value_fc = MLP(
            self.reduced_channels * input_dim * input_dim,
            self.fc_hidden_dim,
            self.support_size,
            self.fc_layers,
        )
        self.policy_conv = nn.Conv2d(
            self.hidden_dim, self.reduced_channels, kernel_size=1
        )
        self.policy_fc = MLP(
            self.reduced_channels * input_dim * input_dim,
            self.fc_hidden_dim,
            num_actions,
            self.fc_layers,
        )

        # for the categorical representation
        self.supports = torch.arange(
            self.support_size, dtype=torch.float32, device=device
        )
        self.supports = torch.pow(2, self.supports)

        if self.device == "cuda":
            self.cuda()

    def initial_inference(self, image, player, numpy=True) -> alpha_zero.NetworkOutput:
        if numpy:
            image = np.expand_dims(image, axis=0)

        if not self.training:
            with torch.no_grad():
                hidden = self.representation_network(image)
                value, policy = self.prediction_network(hidden["hidden_state"])
        else:
            hidden = self.representation_network(image)
            value, policy = self.prediction_network(hidden["hidden_state"])
        if numpy:
            value = self.categorical_to_scalar(torch.softmax(value, dim=1))
            value = value.squeeze(0).cpu().detach().numpy()
            reward = np.array(1, dtype=np.float32)
            policy = policy.half().squeeze(0).cpu().detach().numpy()
        else:
            reward = torch.full(
                (image.shape[0], self.support_size),
                -10,
                dtype=torch.float32,
                device=self.device,
            )
            reward[:, 0] = 1
        return alpha_zero.NetworkOutput(value, reward, policy, hidden)

    def recurrent_inference(
        self, hidden_state, action, numpy=True
    ) -> alpha_zero.NetworkOutput:
        if numpy:
            action = np.expand_dims(action, axis=0)

        if not self.training:
            with torch.no_grad():
                reward, hidden = self.dynamics_network(hidden_state, action)
                value, policy = self.prediction_network(hidden["hidden_state"])
        else:
            reward, hidden = self.dynamics_network(hidden_state, action)
            value, policy = self.prediction_network(hidden["hidden_state"])

        if numpy:
            value = self.categorical_to_scalar(torch.softmax(value, dim=1))
            value = value.squeeze(0).cpu().detach().numpy()
            reward = self.categorical_to_scalar(torch.softmax(reward, dim=1))
            reward = reward.squeeze(0).cpu().detach().numpy()
            policy = policy.half().squeeze(0).cpu().detach().numpy()
        return alpha_zero.NetworkOutput(value, reward, policy, hidden)

    def scale_gradient(self, tensor: torch.Tensor, scale):
        """Scales the gradient for the backward pass."""
        # check if the tensor has a gradient
        if not tensor.requires_grad:
            return tensor
        tensor.register_hook(lambda grad: grad * scale)
        return tensor

    def update_weights(
        self,
        config: alpha_zero.MuZeroConfig,
        optimizer: optim.Optimizer,
        batch,
    ):
        optimizer.zero_grad()
        loss = 0

        image_batch = np.array([data[0] for data in batch])
        actions_batch = [data[1] for data in batch]
        targets_batch = [data[2] for data in batch]
        players_batch = [data[3] for data in batch]

        # Initial step, from the real observation.
        network_output = self.initial_inference(image_batch, None, numpy=False)
        hidden_state = network_output.hidden_state
        # gradient_scale, network_output
        predictions = [network_output]

        # Recurrent steps, from action and previous hidden state.
        for i in range(config.num_unroll_steps):
            actions = []
            for j in range(len(batch)):
                if i < len(actions_batch[j]):
                    actions.append(actions_batch[j][i])
                else:
                    actions.append(
                        alpha_zero.Action(np.random.choice(config.action_space_size))
                    )

            network_output = self.recurrent_inference(
                hidden_state, actions, numpy=False
            )
            hidden_state = network_output.hidden_state
            predictions.append(network_output)

        # Compute the loss using the predictions and the targets.
        for i in range(len(predictions)):
            network_output = predictions[i]
            target_value = [target[i][0] for target in targets_batch]
            target_reward = [target[i][1] for target in targets_batch]
            target_policy = [target[i][2] for target in targets_batch]
            target_masks = [target[i][3] for target in targets_batch]

            target_value = torch.tensor(
                target_value, dtype=torch.float32, device=self.device
            )
            target_reward = torch.tensor(
                target_reward, dtype=torch.float32, device=self.device
            )
            target_policy = torch.tensor(
                target_policy, dtype=torch.float32, device=self.device
            )

            target_value = self.scalar_to_categorical(target_value)
            target_reward = self.scalar_to_categorical(target_reward)

            # calculate the loss with masks
            target_masks = torch.BoolTensor(target_masks).to(self.device)
            l = F.cross_entropy(network_output.value, target_value)
            if i != 0:
                l += F.cross_entropy(network_output.reward, target_reward)
            l += F.cross_entropy(
                network_output.policy_logits[target_masks],
                target_policy[target_masks],
            )

            if i != 0:
                l *= 1 / config.num_unroll_steps
            loss += l

        loss.backward()
        optimizer.step()
        return loss.item()

    def get_weights(self):
        self.cpu()
        weight = {}
        weight["state_dict"] = self.state_dict()
        weight["n_training_steps"] = self.n_training_steps
        return weight

    def set_weights(self, weights):
        self.n_training_steps = weights["n_training_steps"]
        try:
            if self.device == "cuda":
                state_dict = {k: v.cuda() for k, v in weights["state_dict"].items()}
            else:
                state_dict = weights["state_dict"]
            self.load_state_dict(state_dict)
        except Exception as e:
            print("Failed to load weights")
            pass

    def representation_network(self, images):
        hidden_state = torch.tensor(
            images.copy(), dtype=torch.int32, device=self.device
        )

        hidden_state_expanded = hidden_state.unsqueeze(1)
        hidden_state = (hidden_state_expanded == self.power_of_two).float()

        hidden_state = self.representation_conv(hidden_state)
        hidden_state = self.representation_res_blocks(hidden_state)
        return {"hidden_state": hidden_state, "images": images}

    def prediction_network(self, x):
        x = self.prediction_res_blocks(x)

        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = self.value_fc(value)

        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        return value, policy

    def dynamics_network(self, hidden_state, actions):
        new_images = self.step(hidden_state["images"], actions)
        hidden_image = torch.tensor(new_images, dtype=torch.int32, device=self.device)
        hidden_image_expanded = hidden_image.unsqueeze(1)
        hidden_image_expanded = (hidden_image_expanded == self.power_of_two).float()

        hidden = torch.cat((hidden_state["hidden_state"], hidden_image_expanded), dim=1)
        hidden = self.dynamic_conv(hidden)
        hidden = self.dynamic_res_blocks(hidden)

        reward = self.reward_conv(hidden)
        reward = reward.view(reward.size(0), -1)
        reward = self.reward_fc(reward)
        return reward, {
            "hidden_state": hidden,
            "images": new_images,
        }

    def step(self, images, actions):
        new_images = images.copy()
        for i in range(len(images)):
            if actions[i].index == 0:
                reward, new_board = self.merge_up(images[i])
            elif actions[i].index == 1:
                reward, new_board = self.merge_up(np.rot90(images[i], 2))
                new_board = np.rot90(new_board, 2)
            elif actions[i].index == 2:
                reward, new_board = self.merge_left(images[i])
            elif actions[i].index == 3:
                reward, new_board = self.merge_left(np.fliplr(images[i]))
                new_board = np.fliplr(new_board)
            new_images[i] = new_board

        return new_images

    def merge_left(self, board):
        reward = 1
        new_board = np.zeros_like(board)
        for i in range(self.input_dim):
            j = 0
            k = 0
            last_value = 0
            while j < self.input_dim:
                if board[i, j] == 0:
                    j += 1
                else:
                    if last_value == board[i, j]:
                        reward += last_value
                        new_board[i, k - 1] = 2 * last_value
                        last_value = 0
                        j += 1
                    else:
                        new_board[i, k] = board[i, j]
                        last_value = board[i, j]
                        j += 1
                        k += 1

        return reward, new_board

    def merge_up(self, board):
        reward, new_board = self.merge_left(board.T)
        return reward, new_board.T

    def scalar_to_categorical(self, x):
        x_scaled = torch.log2(x)
        # if x is 0, then x_scaled is -inf, so we need to replace it with 0
        x_scaled[x == 0] = 0
        x_scaled_floor = torch.floor(x_scaled)

        # Find the indices of the two nearest supports
        lower_bound = x_scaled_floor.long()
        upper_bound = lower_bound + 1

        # Calculate weights for the nearest supports
        weight_upper = x_scaled - x_scaled_floor
        weight_upper = torch.pow(2, weight_upper) - 1
        weight_lower = 1 - weight_upper

        # Create a categorical representation with zeros and weights for the nearest supports
        categorical_representation = torch.zeros(
            x.size(0), self.support_size, device=self.device
        )
        categorical_representation.scatter_(
            1, lower_bound.unsqueeze(1), weight_lower.unsqueeze(1)
        )
        categorical_representation.scatter_(
            1, upper_bound.unsqueeze(1), weight_upper.unsqueeze(1)
        )

        return categorical_representation

    def categorical_to_scalar(self, x):
        x = torch.matmul(x, self.supports)
        return x

    def compile(self):
        self.representation_network = torch.compile(self.representation_network)
        self.prediction_network = torch.compile(self.prediction_network)
        self.dynamics_network = torch.compile(self.dynamics_network)
        self.scalar_to_categorical = torch.compile(self.scalar_to_categorical)
        self.categorical_to_scalar = torch.compile(self.categorical_to_scalar)


class ReplayBuffer2048(alpha_zero.ReplayBuffer):
    def __init__(self, config: alpha_zero.MuZeroConfig):
        super().__init__(config)

    def sample_game(self, n: int) -> alpha_zero.Game:
        # Sample game with priority given to games with a greater number of moves.
        prob = [len(self.buffer[i].history) for i in range(len(self.buffer))]
        prob = np.array(prob)
        prob = np.power(prob, 1.3)
        prob = prob / sum(prob)
        return np.random.choice(self.buffer, n, p=prob)

    def sample_position(self, games) -> int:
        return [(g, self.sample_position_one(g)) for g in games]

    def sample_position_one(self, game) -> int:
        # Sample position with priority given to recent moves.
        max_pos = len(game.history) - 1
        prob = np.arange(0, max_pos + 1)
        prob = np.power(prob, 0.2)
        prob = prob / sum(prob)
        return np.random.choice(max_pos + 1, p=prob)


def make_network(config: alpha_zero.MuZeroConfig, training: bool) -> alpha_zero.Network:
    network = Network2048(
        BOARD_SIZE, config.action_space_size, training, alpha_zero.DEVICE
    )
    if not training:
        network.eval()
    return network


alpha_zero.Game = Game2048
alpha_zero.Action = Action2048
alpha_zero.ReplayBuffer = ReplayBuffer2048
alpha_zero.Network = Network2048
alpha_zero.make_network = make_network


def visit_softmax_temperature(num_moves, training_steps):
    return 0.5


def make_2048_config() -> alpha_zero.MuZeroConfig:
    config = alpha_zero.MuZeroConfig(
        action_space_size=4,
        max_moves=8192,
        discount=1.0,
        dirichlet_alpha=0.3,
        num_simulations=800,
        batch_size=2048,
        td_steps=8192,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=0.02,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=alpha_zero.KnownBounds(0, 8192),
    )

    # parameters for training
    config.save_path = "pretrained/muzero_2048.pkl"
    config.optimizer = "Adam"
    config.lr_init = 0.001  # Initial learning rate
    config.batch_size = 2048  # Number of simultaneous games in a training batch.
    config.training_steps = 20  # How many batches to learn from (for each training).
    config.batch_training_steps = 10  # Number of training steps in each batch
    config.window_size = 10000  # save latest 1000 games
    config.num_iterations = 1000000  # Total number of games to train on
    config.num_actors = 10  # Number of self-play games before training each network
    config.num_unroll_steps = 5  # Number of game moves unrolled for training
    config.num_workers = 30  # Number of workers to collect data

    # parameters for the game
    config.max_moves = 8192  # Maximum number of moves in a game
    config.td_steps = config.max_moves
    config.action_space_size = BOARD_SIZE  # Size of the action space
    config.known_bounds = alpha_zero.KnownBounds(0, 8192)  # Known bounds of the reward

    # parameters for the self-play
    config.pb_c_init = 0.1  # Initial value of c for the UCB formula
    config.pb_c_base = 19652  # Base value of c for the UCB formula
    config.exploration_constant = 0.0  # Exploration constant

    config.root_dirichlet_alpha = (
        0.3  # Alpha parameter of Dirichlet noise for the root action
    )
    config.root_exploration_fraction = (
        0.25  # Fraction of the exploration budget to be used in the root
    )
    config.num_simulations = 64  # Number of MCTS simulations per move
    return config


if __name__ == "__main__":
    config = make_2048_config()
    if len(sys.argv) > 1:
        alpha_zero.train_muzero(config)
    else:
        alpha_zero.play_muzero(config)
