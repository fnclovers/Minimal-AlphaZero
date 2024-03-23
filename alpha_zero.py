# Lint as: python3
"""Pseudocode description of the MuZero algorithm.
https://github.com/fnclovers/Minimal-AlphaZero"""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test

import collections
import typing
import os
from typing import Any, Dict, List, Optional

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
import pickle as pkl
import os
import torch.multiprocessing as mp
import time
import resource

##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float("inf")
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
torch.set_num_threads(1)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
mp.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")

KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MuZeroConfig:

    def __init__(
        self,
        action_space_size: int,
        max_moves: int,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        batch_size: int,
        td_steps: int,
        num_actors: int,
        lr_init: float,
        lr_decay_steps: float,
        visit_softmax_temperature_fn,
        known_bounds: Optional[KnownBounds] = None,
    ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.exploration_constant = 1

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.save_path = "muzero.pkl"
        self.training_steps = int(1000)
        self.batch_training_steps = int(10)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.optimizer = "SGD"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self):
        return Game(self.action_space_size, self.discount)


def make_board_game_config(
    action_space_size: int, max_moves: int, dirichlet_alpha: float, lr_init: float
) -> MuZeroConfig:

    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=800,
        batch_size=2048,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1),
    )


def make_go_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01
    )


def make_chess_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1
    )


def make_shogi_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1
    )


def make_atari_config() -> MuZeroConfig:

    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
    )


class Action:

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

    def __index__(self):
        return self.index


class Player:
    def __eq__(self, other):
        return True


class Node:

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ActionHistory:
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self, state_index: int = None) -> Player:
        return Player()


class Environment:
    """The environment MuZero is interacting with."""

    def step(self, action):
        pass


class Game:
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment()  # Game specific environment.
        # history of prev actions; used for recurrent inference for training
        self.history = []
        # rewards of prev actions; used for training dynamics network
        self.rewards = []
        # child visit probabilities; used for training policy network
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        # Game specific termination rules.
        pass

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
        return []

    def apply(self, action: Action):
        reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        # Game specific feature planes.
        return []

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        """ The value target is the discounted root value of the search tree N steps
        into the future, plus the discounted sum of all rewards until then.
        """
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += (
                    reward * self.discount**i
                )  # pytype: disable=unsupported-operands

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                # 1) image[n] --> pred[n], value[n], hidden_state[n]
                # 2) hidden_state[n] + action[n] --> reward[n], pred[n+1], value[n+1], hidden_state[n+1]
                targets.append(
                    (value, last_reward, self.child_visits[current_index], True)
                )
            else:
                # States past the end of games are treated as absorbing states.
                targets.append(
                    (value, last_reward, [0] * self.action_space_size, False)
                )
        return targets

    def to_play(self, state_index: int = None) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    def print_game(self, state_index: int):
        pass

    def get_score(self, state_index: int):
        return len(self.history)


class ReplayBuffer:

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size :]
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = self.sample_game(self.batch_size)
        game_pos = self.sample_position(games)
        batch = [
            (
                g.make_image(i),
                g.history[i : i + num_unroll_steps],
                g.make_target(i, num_unroll_steps, td_steps),
                g.to_play(i),
            )
            for (g, i) in game_pos
        ]
        return batch

    def sample_game(self, n: int) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return np.random.choice(self.buffer, n)

    def sample_position(self, games) -> int:
        # Sample position from game either uniformly or according to some priority.
        return [(g, np.random.choice(len(g.history))) for g in games]


class NetworkOutput(typing.NamedTuple):
    value: np.ndarray
    reward: np.ndarray
    policy_logits: np.ndarray
    hidden_state: Any


class Network:
    def __init__(self):
        self.n_training_steps = 0

    def initial_inference(self, image, player) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0, 0, {}, [])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def set_weights(self, weights):
        # Sets the weights of this network.
        pass

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.n_training_steps

    def increment_training_steps(self):
        self.n_training_steps += 1

    def update_weights(
        self,
        config: MuZeroConfig,
        optimizer: optim.Optimizer,
        batch,
    ):
        # Update the weights of this network given a batch of data.
        return 0


def make_network(config: MuZeroConfig, training: bool) -> Network:
    return Network()


class SharedStorage:

    def __init__(self, config: MuZeroConfig):
        self.config = config
        self._weights = {}

    def latest_network(self, training=True) -> Network:
        if self._weights:
            new_network = make_network(self.config, training=training)
            new_network.set_weights(self._weights[max(self._weights.keys())])
            return new_network
        else:
            return make_network(self.config, training=training)

    def save_network(self, step: int, weights):
        self._weights = {}
        self._weights[step] = weights


##### End Helpers ########
##########################


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        run_selfplay(config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
    while True:
        network = storage.latest_network(training=False)
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        game, action = predict_action(config, network, game)
        game.apply(action)
    return game


def predict_action(
    config: MuZeroConfig, network: Network, game: Game, print: bool = False
) -> Game:
    min_max_stats = MinMaxStats(config.known_bounds)

    root = Node(0)
    current_observation = game.make_image(-1)
    network_output = network.initial_inference(current_observation, game.to_play())
    expand_node(root, game.to_play(), game.legal_actions(), network_output)
    backpropagate(
        [root],
        network_output.value,
        game.to_play(),
        config.discount,
        min_max_stats,
    )
    add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using only action sequences and the
    # model learned by the network.
    run_mcts(config, root, game.action_history(), network, min_max_stats)
    action = select_action(config, len(game.history), root, network)
    game.store_search_statistics(root)
    if print:
        print_node(config, game, root, min_max_stats, max_depth=2)
    return game, action


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
    config: MuZeroConfig,
    root: Node,
    action_history: ActionHistory,
    network: Network,
    min_max_stats: MinMaxStats,
):
    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(
            parent.hidden_state, history.last_action()
        )
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount,
            min_max_stats,
        )


def softmax_sample(distribution, temperature: float):
    # if temperature is 0, then we always select the action with the highest probability
    if temperature == 0:
        index = np.argmax([p for p, _ in distribution])
        return distribution[index]

    policy = np.array([p for p, _ in distribution])
    policy = np.exp((policy - policy.max()) / temperature)
    policy /= policy.sum()
    index = np.random.choice(len(distribution), p=policy)
    return distribution[index]


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: Network):
    # if True:
    #     # select the action with the highest value
    #     return max(node.children.items(), key=lambda act_node: act_node[1].value())[0]

    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps()
    )
    _, action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    _, action, child = max(
        (sum(ucb_score(config, node, child, min_max_stats)), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(
    config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats
) -> float:
    pb_c = (
        np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = child.reward + config.discount * child.value()
        if parent.to_play != child.to_play:
            value_score = -value_score
        value_score = min_max_stats.normalize(value_score)
    else:
        value_score = min_max_stats.normalize(parent.value())

    exploration_score = config.exploration_constant * np.sqrt(
        2 * np.log(parent.visit_count) / (child.visit_count + 1)
    )
    return prior_score, value_score, exploration_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(
    node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput
):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = [network_output.policy_logits[actions[i]] for i in range(len(actions))]
    policy = np.array(policy, dtype=np.float32)
    # remove infinities
    policy[policy > 10000] = 10000
    policy[policy < -10000] = -10000
    policy = np.exp(policy - np.max(policy))
    policy_sum = np.sum(policy)
    for i in range(len(actions)):
        node.children[actions[i]] = Node(policy[i] / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
    search_path: List[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        # value = curr_reward + discount * next_reward + discount^2 * next_next_reward + ...
        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# Prints the information of a MCTS node for debugging purposes
def print_node(
    config: MuZeroConfig,
    game: Game,
    parent: Node,
    min_max_stats: MinMaxStats,
    child: Node = None,
    depth: int = 0,
    string: str = "",
    max_depth: int = 3,
):
    if depth > max_depth:
        return

    if child is not None:
        prior_score, value_score, exploration_score = ucb_score(
            config, parent, child, min_max_stats
        )
        # print Node ||| MTCS Stat (value, visit_count, prior) ||| UCB Score (prior_score, value_score, exploration_score)
        print(
            f"{' ' * depth}Node[{string}]: {child.value():.2f} | {child.visit_count} | {100 * child.prior:.2f} ||| {100 * prior_score:.2f} | {100 * value_score:.2f} | {100 * exploration_score:.2f}"
        )
        parent = child
    else:
        # root node
        game.print_game(-1)
        print(f"{' ' * depth}Node: {parent.value():.2f} | {parent.visit_count}")

    for action, child in parent.children.items():
        new_string = string
        if len(new_string) > 0:
            new_string += ">"
        new_string += str(action)
        if child.expanded():
            print_node(
                config,
                game,
                parent,
                min_max_stats,
                child,
                depth + 1,
                new_string,
                max_depth,
            )


# Main function to play MuZero
def play_muzero(config: MuZeroConfig, selfplay: bool = False):
    # load the latest network, games
    if os.path.exists(config.save_path):
        with open(config.save_path, "rb") as f:
            data = pkl.load(f)
            storage = data["storage"]
            replay_buffer = data["replay_buffer"]
            games_updated = data["games_updated"]
            games_trained = data["games_trained"]
            # replay_buffer.buffer = []
            print(
                f"Loaded from {config.save_path}: {games_updated} games updated, {games_trained} games trained"
            )

        # see how many score the last game got
        last_game = replay_buffer.buffer[-1]
        for i in range(len(last_game.history)):
            print(f"Action probabilities: ", end="")
            action_prob = last_game.child_visits[i]
            for j in range(config.action_space_size):
                print(f"{action_prob[j] * 100:.2f}", end=" ")
            print()
            last_game.print_game(i)
    else:
        storage = SharedStorage(config)
        replay_buffer = ReplayBuffer(config)

    network = storage.latest_network(training=False)
    game = config.new_game()
    score = 0
    while not game.terminal():
        network_output = network.initial_inference(game.make_image(-1), game.to_play())
        game, action = predict_action(config, network, game, True)
        net_prob = np.exp(
            network_output.policy_logits - np.max(network_output.policy_logits)
        )
        net_prob = net_prob / np.sum(net_prob)

        print(f"Game score: {score}")
        print(f"Value: {game.root_values[-1]:.2f}({network_output.value:.2f})")
        print(f"Action probabilities: ", end="")
        action_prob = game.child_visits[-1]
        for i in range(config.action_space_size):
            print(
                f"{action_prob[i] * 100:.2f}({i + 1}, {net_prob[i] * 100:.2f})", end=" "
            )
        print()
        game.print_game(-1)
        print()

        if selfplay:
            next_action = action
        else:
            while True:
                try:
                    next_action = input(
                        f"Enter action (1 ~ {config.action_space_size}): "
                    )
                    if next_action == "q":
                        os._exit(0)
                    next_action = int(next_action) - 1
                    if not 0 <= next_action < config.action_space_size:
                        raise ValueError
                    next_action = Action(next_action)
                    break
                except:
                    print("Invalid input")
                    pass

        game.apply(next_action)
        score += game.rewards[-1]

    exit()


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: MuZeroConfig, network: Network, replay_buffer: ReplayBuffer):
    training_steps = network.training_steps()
    learning_rate = config.lr_init * config.lr_decay_rate ** (
        training_steps / config.lr_decay_steps
    )
    if config.optimizer == "SGD":
        optimizer = optim.SGD(
            network.parameters(),
            lr=learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=config.weight_decay,
        )

    avg_loss = 0
    for _ in range(config.training_steps):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        for _ in range(config.batch_training_steps):
            avg_loss += network.update_weights(config, optimizer, batch)
    print(f"Loss: {avg_loss / config.training_steps / config.batch_training_steps}")
    return network


def play_game_processor(GAME_QUEUE, config, network):
    start_time = time.time()
    game = play_game(config, network)
    end_time = time.time()
    print(f"Game time: {end_time - start_time:.2f}")
    GAME_QUEUE.put(game)

    # wait until queue is empty
    GAME_QUEUE.join()


def train_network_processor(NETWORK_QUEUE, config, network, replay_buffer):
    start_time = time.time()
    network = train_network(config, network, replay_buffer)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f}")
    network.cpu()
    NETWORK_QUEUE.put(network.get_weights())

    # wait until queue is empty
    NETWORK_QUEUE.join()


def train_muzero(config: MuZeroConfig):
    global USE_GPU, DEVICE

    games_updated = 0
    games_trained = 0

    if os.path.exists(config.save_path):
        with open(config.save_path, "rb") as f:
            data = pkl.load(f)
            storage = data["storage"]
            replay_buffer = data["replay_buffer"]
            games_updated = data["games_updated"]
            games_trained = data["games_trained"]
            print(
                f"Loaded from {config.save_path}: {games_updated} games updated, {games_trained} games trained"
            )
    else:
        storage = SharedStorage(config)
        replay_buffer = ReplayBuffer(config)

    # use multiple processes to collect training data
    mp.set_start_method("spawn")
    GAME_QUEUE = mp.JoinableQueue(128 * 1024 * 1024)
    TRAINING_QUEUE = mp.JoinableQueue(128 * 1024 * 1024)
    N_CPU = config.num_workers
    game_processes = []
    training_processes = None
    ended = False

    while not ended:
        # check if any processes have finished
        idx = 0
        while len(game_processes) > 0 and idx < len(game_processes):
            if not game_processes[idx].is_alive():
                game_processes[idx].join()
                # destroy the process
                game_processes[idx].close()
                del game_processes[idx]
            else:
                idx += 1

        # start new processes
        while len(game_processes) < N_CPU and games_updated < config.num_iterations:
            # self-play on CPU
            USE_GPU = False
            DEVICE = "cuda" if USE_GPU else "cpu"
            network = storage.latest_network(training=False)
            p = mp.Process(
                target=play_game_processor,
                args=(GAME_QUEUE, config, network),
            )
            p.start()
            game_processes.append(p)

        while not GAME_QUEUE.empty():
            game = GAME_QUEUE.get()
            replay_buffer.save_game(game)
            games_updated += 1
            print(f"Game {games_updated} score: {game.get_score(-1)}")
            GAME_QUEUE.task_done()

        if (
            games_updated >= games_trained + config.num_actors
            or games_updated >= config.num_iterations
        ):
            if training_processes is None or not training_processes.is_alive():
                # train network on GPU
                print(f"Training network {games_updated}")
                USE_GPU = True
                DEVICE = "cuda" if USE_GPU else "cpu"
                network = storage.latest_network(training=True)
                training_processes = mp.Process(
                    target=train_network_processor,
                    args=(TRAINING_QUEUE, config, network, replay_buffer),
                )
                training_processes.start()
                games_trained = games_updated

        while not TRAINING_QUEUE.empty():
            weights = TRAINING_QUEUE.get()
            storage.save_network(games_trained, weights)
            TRAINING_QUEUE.task_done()
            with open(config.save_path, "wb") as f:
                pkl.dump(
                    {
                        "storage": storage,
                        "replay_buffer": replay_buffer,
                        "games_updated": games_updated,
                        "games_trained": games_trained,
                    },
                    f,
                )
            print(f"Trained network {games_trained}")
            if games_trained >= config.num_iterations:
                ended = True

        time.sleep(1)


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################
