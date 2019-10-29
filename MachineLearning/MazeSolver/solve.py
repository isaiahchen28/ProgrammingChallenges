'''
Maze Solver

Code and explanations adapted from https://samyzaf.com/ML/rl/qmaze.html


Introduction
------------------------------------------------------------------------------
This code uses deep reinforcement learning to solve a maze. Reinforcement
learning is a machine learning technique for solving problems by a feedback
system (rewards and penalties) applied on an agent which operates in an
environment and needs to move through a series of states in order to reach a
pre-defined final state. For example, an agent is trying to find the shortest
route from a starting cell to a target cell in a maze (environment). The agent
is experimenting and exploiting past experiences (episodes) in order to
achieve its goal. It may fail again and again, but hopefully, after lots of
trial and error (rewards and penalties) it will arrive to the solution of the
problem.

The solution will be reached if the agent finds the optimal sequence of states
in which the accumulated sum of rewards is maximal (in short, we lure the
agent to accumulate a maximal reward, and while doing so, it actually solves
our problem). Note that it may happen that in order to reach the goal, the
agent will have to endure many penalties (negative rewards) on its way. For
example, the agent in a maze gets a small penalty for every legal move. The
reason for that is that we want it to get to the target cell in the shortest
possible path. However, the shortest path to the target cell is sometimes long
and winding, and our agent may have to endure many penalties until it gets to
the target (sometimes called "delayed reward").

For this code, the target cell will always be the bottom-right corner of the
maze. The deep learning libraries used involve keras and TensorFlow. Matplotlb
is used to generate images.


Rewarding Scheme
------------------------------------------------------------------------------
Our rewards will be floats ranging from -1.0 to 1.0.

Each move from one state to the next state will be rewarded (the agent gets
points) by a positive or a negative (penalty) amount.

Each move from one cell to an adjacent cell will cost the agent -0.04 points.
This should discourage the agent from wandering around and get to the target
in the shortest route possible.

The maximal reward of 1.0 points is given when the agent reaches the target.

An attempt to enter a blocked cell will cost the agent -0.75 points! This is a
severe penalty, so hopefully the agent will learn to avoid it completely. Note
that an attempt to move to a blocked cell is invalid and will not be executed,
but it will incur a -0.75 penalty if attempted.

The same rule holds for an attempt to move outside the maze boundaries, with a
slightly higher penalty of -0.8 points.

The agent will be penalized by -0.25 points for any move to a cell which has
already been visited.

To avoid infinite loops and senseless wandering, the game is ended once the
total reward of the agent is below a negative threshold.


Q-learning
------------------------------------------------------------------------------
The main objective of Q-learning is to develop a policy for navigating the
maze successfully. Presumably, after playing hundreds of games, the agent
should attain a clear deterministic policy for how to act in every possible
situation.

The policy is a function that takes a maze snapshot (envstate) as input and
returns the action to be taken by the agent. The input consists of the full
maze state and the location of the agent.

At the start, we simply choose a completely random policy. Then we use it to
play thousands of games from which we learn how to perfect it. Surely, at the
early training stages, our policy will yield lots of errors and cause us to
lose many games, but our rewarding policy should provide feedback for it on
how to improve itself. The learning engine is going to be a simple
feed-forward neural network which takes an environment state as input and
yields a reward per action vector.

There are two types of moves in regard to Q-learning:
    Exploitation: these are moves that our policy dictates based on previous
                  experiences. The policy function is used in about 90% of the
                  moves before it is completed.
    Exploration: in about 10% of the cases, we take a completely random action
                 in order to acquire new experiences (and possibly meet bigger
                 rewards) which our strategy function may not allow us to make
                 due to its restrictive nature.

The exploration factor, epsilon, is the the frequency level of how much
exploration to do. It is usually set to 0.1, which roughly means that in one
of every 10 moves the agent takes a completely random action.
'''
from generate import load_maze
import datetime
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_actions():
    '''
    Dictionary reprenting possible actions to take in the maze.
    **Returns**
        action_map: *dict, int, str*
            A dictionary that will correlate the integer key to
            a possible action.
    '''
    # Define actions
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    # Return the dictionary of possible actions
    return {
        LEFT: 'left',
        UP: 'up',
        RIGHT: 'right',
        DOWN: 'down',
    }


class Qmaze(object):
    '''
    Class object for representing the maze in the context of Q-learning.
    **Parameters**
        maze: *list, list, floats* or *numpy array*
            A data structure that represents valid paths, walls, and current
            positions.
        pos: *tuple, int*
            The current position of the pointer.
    '''

    def __init__(self, maze, pos=(0, 0)):
        # Save maze as a NumPy array if it is not already a NumPy array
        self.maze = np.array(maze)
        # Define the target position
        nrows, ncols = self.maze.shape
        self.target = (nrows - 1, ncols - 1)
        # Define valid paths
        self.paths = [(r, c) for r in range(nrows)
                      for c in range(ncols) if self.maze[r, c] == 1.0]
        # Remove the target position from the list of valid paths
        self.paths.remove(self.target)
        # Checks for if a maze is valid or not
        if self.maze[self.target] == 0.0:
            raise Exception("Invalid Maze: target cell cannot be blocked!")
        if pos not in self.paths:
            raise Exception("Invalid Location")
        # Call reset function during initialization
        self.reset(pos)

    def reset(self, pos):
        '''
        Reset the pointer to a given position and update appropriate
        variables.
        '''
        self.pos = pos
        self.maze = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        row, col = pos
        # Set maze position
        self.maze[row, col] = 0.5
        self.state = (row, col, 'start')
        # Define the minimum reward and initialize total reward
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        # Initialize set of visited spaces
        self.visited = set()

    def update_state(self, action):
        '''
        Update the state of the maze given a specified action to take.
        '''
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = pos_row, pos_col, mode = self.state
        # Mark a visited space by adding it to the appropriate set
        if self.maze[pos_row, pos_col] > 0.0:
            self.visited.add((pos_row, pos_col))
        # Obtain list of valid actions
        valid_actions = self.valid_actions()
        # Check if there are any valid actions and adjust position accordingly
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == 0:
                ncol -= 1
            elif action == 1:
                nrow -= 1
            if action == 2:
                ncol += 1
            elif action == 3:
                nrow += 1
        # If we reach this point, there is an invalid action and we keep the
        # same position
        else:
            nmode = 'invalid'
        # Define the new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        '''
        Retrieve the appropriate reward value as described in the rewarding
        scheme above.
        '''
        # Obtain position
        pos_row, pos_col, mode = self.state
        nrows, ncols = self.maze.shape
        # Check if we reached the target
        if pos_row == nrows - 1 and pos_col == ncols - 1:
            return 1.0
        # Check if our action is blocked
        if mode == 'blocked':
            return self.min_reward - 1
        # Check if we visited this position before
        if (pos_row, pos_col) in self.visited:
            return -0.25
        # Check if our action is invalid
        if mode == 'invalid':
            return -0.75
        # Check if our action is valid
        if mode == 'valid':
            return -0.04

    def act(self, action):
        '''
        Given a specific action, update the state of the maze and obtain the
        appropriate reward value.
        '''
        # Update state
        self.update_state(action)
        # Get reward and update the total reward
        reward = self.get_reward()
        self.total_reward += reward
        # Obtain win or loss status
        status = self.game_status()
        # Get the current environment
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        '''
        Obtain the current environment of the maze.
        '''
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        '''
        Draw the maze environment.
        '''
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # Clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # Draw the current pointer
        row, col, valid = self.state
        canvas[row, col] = 0.5
        return canvas

    def game_status(self):
        '''
        Check the status of the game.
        '''
        if self.total_reward < self.min_reward:
            return 'lose'
        pos_row, pos_col, mode = self.state
        nrows, ncols = self.maze.shape
        if pos_row == nrows - 1 and pos_col == ncols - 1:
            return 'win'
        # If we reach this point, the game is still in progress
        return 'not_over'

    def valid_actions(self, cell=None):
        '''
        Get a list of valid actions based on the current position and nearby
        environment.
        '''
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        # Define the list of possible actions:
        #   LEFT = 0
        #   UP = 1
        #   RIGHT = 2
        #   DOWN = 3
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        # Remove UP or DOWN
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)
        # Remove LEFT or RIGHT
        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)
        # Remove UP or DOWN
        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(3)
        # Remove LEFT or RIGHT
        if col > 0 and self.maze[row, col - 1] == 0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(2)
        return actions


def show(qmaze):
    '''
    Show the current state of the qmaze object.
    '''
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    # Define current position
    canvas[rat_row, rat_col] = 0.3
    # Define target position
    canvas[nrows - 1, ncols - 1] = 0.9
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.show()
    return img


def play_game(model, qmaze, pos):
    '''
    Play the game given a neural network, a maze, and the starting position.
    '''
    # Reset to the initial position if not there already
    qmaze.reset(pos)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # Get the next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        # Apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    '''
    Check if the maze is completed or not.
    '''
    for cell in qmaze.paths:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


class Experience(object):
    '''
    Class object for collecting game episodes/experiences in a list.
    **Parameters**
        model:
            The neural network model to be used.
        max_memory: *int*
            The maximum number of episodes to keep. When the maximum is
            reached, the oldest episode is to be removed.
        discount: *float*
             A special coefficient required for the Bellman equation for
             stochastic environments.
    '''

    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        '''
        Add the latest episode to the memory and remove the oldest if
        necessary.
        '''
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        '''
        Call the predict function from the model.
        '''
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        '''
        Retrieve relevant data
        '''
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size,
                                               replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


def qtrain(model, maze, epsilon=0.1, **opt):
    '''
    Train the neural network model needed to solve the maze.
    '''
    # Define the number of training epochs
    n_epoch = opt.get('n_epoch', 15000)
    # Define the maximum number of game experiences kept in memory
    max_memory = opt.get('max_memory', 1000)
    # Define the number of samples used in each training epoch
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()
    # If you want to continue training from a previous model, make sure the h5
    # file is located in the same directory
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)
    # Construct environment/game from the maze
    qmaze = Qmaze(maze)
    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)
    # Initialize the history of winning/losing games
    win_history = []
    # Define the window size of the history
    hsize = qmaze.maze.size // 2
    # Initialize win rate
    win_rate = 0.0
    for epoch in range(n_epoch):
        # Initialize values for the beginning of each epoch
        loss = 0.0
        pos = random.choice(qmaze.paths)
        qmaze.reset(pos)
        game_over = False
        # Get initial envstate
        envstate = qmaze.observe()
        n_episodes = 0
        # Play the game
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions:
                break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))
            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False
            # Store the episode in memory
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1
            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            loss = model.evaluate(inputs, targets, verbose=0)
        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
        # Calculate current time
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        # Print results to terminal
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch - 1, loss,
                              n_episodes, sum(win_history), win_rate, t))
        # Check if training has exhausted all free cells and if in all cases
        # the agent won
        if win_rate > 0.9:
            epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model,
                                                                   qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break
    # Save the trained model weights and architecture
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    # Print final results
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" %
          (epoch, max_memory, data_size, t))
    return seconds


def format_time(seconds):
    '''
    Format the time output so that it prints nicely.
    '''
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


def build_model(maze):
    '''
    Build the neural network model
    '''
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(49, input_shape=(49,), activation='relu'))
    model.add(tf.keras.layers.Dense(49, activation='relu'))
    model.add(tf.keras.layers.Dense(4))
    model.compile(optimizer='RMSprop', loss='mse')
    return model


if __name__ == '__main__':
    maze = load_maze("maze")
    maze = np.array([[float(j) for j in i] for i in maze])
    model = build_model(maze)
    qtrain(model, maze)
