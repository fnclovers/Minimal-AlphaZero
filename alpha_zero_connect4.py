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

BOARD_SIZE = 7


class ActionConnect4(alpha_zero.Action):

    def __init__(self, index: int):
        super().__init__(index)

    def __str__(self) -> str:
        return str(self.index)


class PlayerConnect4(object):
    def __init__(self, player: int):
        self.player = player

    def __eq__(self, other):
        return self.player == other.player

    def __str__(self) -> str:
        return str(self.player)


# redefine the game
class GameConnect4(alpha_zero.Game):
    def __init__(self, action_space_size: int, discount: float):
        super().__init__(action_space_size, discount)
        self.size = BOARD_SIZE
        board = np.zeros((self.size, self.size), dtype=np.int32)
        self.boards = [board]  # 1 for player 1, 2 for player 2
        self.is_terminal = False

    def terminal(self) -> bool:
        return self.is_terminal

    def legal_actions(self) -> List[alpha_zero.Action]:
        board = self.boards[-1]
        moves = []
        for i in range(self.size):
            if board[0, i] == 0:
                moves.append(ActionConnect4(i))
        return moves

    def apply(self, action: alpha_zero.Action):
        board = self.boards[-1]
        new_board = board.copy()
        curr_turn = len(self.history) % 2

        for i in range(self.size - 1, -1, -1):
            if new_board[i, action.index] == 0:
                new_board[i, action.index] = curr_turn + 1
                break

        status = self.check_status(new_board)
        if status == 1:
            reward = 1
            self.is_terminal = True
        elif status == -1:
            reward = 0
            self.is_terminal = True
        else:
            reward = 0

        self.boards.append(new_board)
        self.rewards.append(reward)
        self.history.append(action)

    def make_image(self, state_index: int):
        return self.boards[state_index]

    def make_target(
        self,
        state_index: int,
        num_unroll_steps: int,
        td_steps: int,
    ):
        winner = (
            len(self.history) - 1
        ) % 2  # 0 for player 1 (first), 1 for player 2 (second)

        # calculate the target
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            curr_turn = current_index % 2
            if curr_turn == winner:
                target_value = self.rewards[-1]
            else:
                target_value = -self.rewards[-1]

            if current_index < len(self.history) - 1:
                targets.append(
                    (target_value, 0, self.child_visits[current_index], True)
                )
            elif current_index == len(self.history) - 1:
                targets.append(
                    (target_value, 0, self.child_visits[current_index], True)
                )
            else:
                targets.append(
                    (
                        target_value,
                        0,
                        [0] * self.action_space_size,
                        False,
                    )
                )

        return targets

    def to_play(self, state_index: int = None) -> alpha_zero.Player:
        if state_index == None:
            state_index = len(self.history)
        return PlayerConnect4(state_index % 2)

    def check_status(self, board):
        # check for horizontal win
        for i in range(self.size):
            for j in range(self.size - 3):
                if (
                    board[i, j] != 0
                    and board[i, j] == board[i, j + 1]
                    and board[i, j] == board[i, j + 2]
                    and board[i, j] == board[i, j + 3]
                ):
                    return 1

        # check for vertical win
        for i in range(self.size - 3):
            for j in range(self.size):
                if (
                    board[i, j] != 0
                    and board[i, j] == board[i + 1, j]
                    and board[i, j] == board[i + 2, j]
                    and board[i, j] == board[i + 3, j]
                ):
                    return 1

        # check for diagonal win
        for i in range(self.size - 3):
            for j in range(3, self.size):
                if (
                    board[i, j] != 0
                    and board[i, j] == board[i + 1, j - 1]
                    and board[i, j] == board[i + 2, j - 2]
                    and board[i, j] == board[i + 3, j - 3]
                ):
                    return 1

        for i in range(self.size - 3):
            for j in range(self.size - 3):
                if (
                    board[i, j] != 0
                    and board[i, j] == board[i + 1, j + 1]
                    and board[i, j] == board[i + 2, j + 2]
                    and board[i, j] == board[i + 3, j + 3]
                ):
                    return 1

        # check for draw
        if np.all(board != 0):
            return -1

        return 0

    def print_game(self, state_index: int) -> str:
        if state_index == -1:
            state_index = len(self.boards) - 1
        board = self.boards[state_index]
        if state_index == 0:
            print(
                f"############### Initial State | Player {PlayerConnect4(state_index % 2)} ###############"
            )
        else:
            print(
                f"########### State {state_index} | Action {self.history[state_index - 1]} | Player {PlayerConnect4(state_index % 2)} | Value {self.root_values[state_index]:.2f} ###########"
            )
        for i in range(self.size):
            # pretty print the board
            row = "|"
            for j in range(self.size):
                if board[i, j] == 0:
                    row += "   |"
                elif board[i, j] == 1:
                    row += " O |"
                else:
                    row += " X |"
            print(row)
        print(f"##############################################")


class ActionHistoryConnect4(alpha_zero.ActionHistory):
    def __init__(self, history, action_space_size):
        super().__init__(history, action_space_size)

    def to_play(self, state_index: int = None):
        if state_index == None:
            state_index = len(self.history)
        return PlayerConnect4(state_index % 2)


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


class NetworkConnect4(nn.Module, alpha_zero.Network):
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
        self.training = training

        # for the representation function
        self.power_of_two = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.power_of_two = self.power_of_two.view(1, -1, 1, 1).to(device)
        self.representation_conv = nn.Conv2d(
            4, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.representation_res_blocks = nn.Sequential(
            *[ResidualBlock(self.hidden_dim) for _ in range(self.representation_layers)]
        )

        # for the dynamics function
        self.dynamic_conv = nn.Conv2d(
            4 + self.hidden_dim,
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
            1,
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
            1,
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

        if self.device == "cuda":
            self.cuda()

    def initial_inference(self, image, player, numpy=True) -> alpha_zero.NetworkOutput:
        if numpy:
            image = np.expand_dims(image, axis=0)

        if not self.training:
            with torch.no_grad():
                hidden = self.representation_network(image, [player])
                value, policy = self.prediction_network(hidden["hidden_state"])
        else:
            hidden = self.representation_network(image, player)
            value, policy = self.prediction_network(hidden["hidden_state"])
        if numpy:
            value = value.squeeze(0).cpu().detach().numpy()
            reward = np.array(0, dtype=np.float32)
            policy = policy.half().squeeze(0).cpu().detach().numpy()
        else:
            reward = torch.full(
                (image.shape[0],),
                0,
                dtype=torch.float32,
                device=self.device,
            )
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
            value = value.squeeze(0).cpu().detach().numpy()
            reward = reward.squeeze(0).cpu().detach().numpy()
            policy = policy.half().squeeze(0).cpu().detach().numpy()
        return alpha_zero.NetworkOutput(value, reward, policy, hidden)

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
        network_output = self.initial_inference(image_batch, players_batch, numpy=False)
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

            # calculate the loss with masks
            target_masks = torch.BoolTensor(target_masks).to(self.device)
            l = F.mse_loss(network_output.value, target_value)
            if i != 0:
                l += F.mse_loss(network_output.reward, target_reward)
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

    def representation_network(self, images, player):
        hidden_state = torch.tensor(
            images.copy(), dtype=torch.int32, device=self.device
        )
        curr_turn = torch.tensor(
            [p.player % 2 for p in player], dtype=torch.bool, device=self.device
        )
        curr_turn = (
            curr_turn.view(-1, 1, 1, 1)
            .expand(-1, 1, self.input_dim, self.input_dim)
            .float()
        )

        hidden_state_expanded = hidden_state.unsqueeze(1)
        hidden_state = (hidden_state_expanded == self.power_of_two).float()
        hidden_state = torch.cat((hidden_state, curr_turn), dim=1)

        hidden_state = self.representation_conv(hidden_state)
        hidden_state = self.representation_res_blocks(hidden_state)
        return {"hidden_state": hidden_state, "images": images, "turn": curr_turn}

    def prediction_network(self, x):
        x = self.prediction_res_blocks(x)

        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = F.tanh(self.value_fc(value))
        value = value.squeeze(-1)

        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        return value, policy

    def dynamics_network(self, hidden_state, actions):
        new_images = self.step(hidden_state["images"], actions, hidden_state["turn"])
        hidden_image = torch.tensor(new_images, dtype=torch.int32, device=self.device)
        hidden_image_expanded = hidden_image.unsqueeze(1)
        hidden_image_expanded = (hidden_image_expanded == self.power_of_two).float()

        curr_turn = (hidden_state["turn"] == 0).float()

        hidden = torch.cat(
            (hidden_state["hidden_state"], hidden_image_expanded, curr_turn), dim=1
        )
        hidden = self.dynamic_conv(hidden)
        hidden = self.dynamic_res_blocks(hidden)

        reward = self.reward_conv(hidden)
        reward = reward.view(reward.size(0), -1)
        reward = F.tanh(self.reward_fc(reward))
        reward = reward.squeeze(-1)
        return reward, {
            "hidden_state": hidden,
            "images": new_images,
            "turn": curr_turn,
        }

    def step(self, images, actions, turn):
        new_images = images.copy()
        for i in range(len(images)):
            curr_turn = turn[i][0][0][0].item()
            for j in range(self.input_dim - 1, -1, -1):
                if new_images[i, j, actions[i].index] == 0:
                    new_images[i, j, actions[i].index] = curr_turn + 1
                    break

        return new_images

    def compile(self):
        self.representation_network = torch.compile(self.representation_network)
        self.prediction_network = torch.compile(self.prediction_network)
        self.dynamics_network = torch.compile(self.dynamics_network)
        self.scalar_to_categorical = torch.compile(self.scalar_to_categorical)
        self.categorical_to_scalar = torch.compile(self.categorical_to_scalar)


class ReplayBufferConnect4(alpha_zero.ReplayBuffer):
    def __init__(self, config: alpha_zero.MuZeroConfig):
        super().__init__(config)

    def sample_game(self, n: int) -> alpha_zero.Game:
        # Sample game from buffer either uniformly or according to some priority.
        return np.random.choice(self.buffer, n)

    def sample_position(self, games) -> int:
        # Sample position from game either uniformly or according to some priority.
        return [(g, np.random.choice(len(g.history))) for g in games]


def make_network(config: alpha_zero.MuZeroConfig, training: bool) -> alpha_zero.Network:
    network = NetworkConnect4(
        BOARD_SIZE, config.action_space_size, training, alpha_zero.DEVICE
    )
    if not training:
        network.eval()
        for param in network.parameters():
            param.grad = None
    return network


alpha_zero.Game = GameConnect4
alpha_zero.Action = ActionConnect4
alpha_zero.Player = PlayerConnect4
alpha_zero.ActionHistory = ActionHistoryConnect4
alpha_zero.ReplayBuffer = ReplayBufferConnect4
alpha_zero.Network = NetworkConnect4
alpha_zero.make_network = make_network


def visit_softmax_temperature(num_moves, training_steps):
    return 1


def make_connect4_config() -> alpha_zero.MuZeroConfig:
    config = alpha_zero.MuZeroConfig(
        action_space_size=4,
        max_moves=512,
        discount=1.0,
        dirichlet_alpha=0.3,
        num_simulations=800,
        batch_size=2048,
        td_steps=512,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=0.02,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=alpha_zero.KnownBounds(-1, 1),
    )

    # parameters for training
    config.save_path = "pretrained/muzero_connect4.pkl"
    config.optimizer = "Adam"
    config.lr_init = 0.001  # Initial learning rate
    config.batch_size = 2048  # Number of simultaneous games in a training batch.
    config.training_steps = 20  # How many batches to learn from (for each training).
    config.batch_training_steps = 10  # Number of training steps in each batch
    config.window_size = 5000  # save latest 1000 games
    config.num_iterations = 1000000  # Total number of games to train on
    config.num_actors = 100  # Number of self-play games before training each network
    config.num_unroll_steps = 5  # Number of game moves unrolled for training
    config.num_workers = 30  # Number of workers to collect data

    # parameters for the game
    config.max_moves = 512  # Maximum number of moves in a game
    config.td_steps = config.max_moves
    config.action_space_size = BOARD_SIZE  # Size of the action space
    config.known_bounds = alpha_zero.KnownBounds(-1, 1)  # Known bounds of the reward

    # parameters for the self-play
    config.pb_c_init = 0.5  # Initial value of c for the UCB formula
    config.pb_c_base = 19652  # Base value of c for the UCB formula
    config.exploration_constant = 0.03  # Exploration constant

    config.root_dirichlet_alpha = (
        0.3  # Alpha parameter of Dirichlet noise for the root action
    )
    config.root_exploration_fraction = (
        0.25  # Fraction of the exploration budget to be used in the root
    )
    config.num_simulations = 64  # Number of MCTS simulations per move
    return config


if __name__ == "__main__":
    config = make_connect4_config()
    if sys.argc > 1:
        alpha_zero.train_muzero(config)
    else:
        alpha_zero.play_muzero(config)
