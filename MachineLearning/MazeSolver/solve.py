'''
Maze Solver
https://samyzaf.com/ML/rl/qmaze.html
'''
from __future__ import print_function
import datetime
import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt


class Qmaze(object):
    '''
    Class object for representing the maze in the context of Q-learning
    **Parameters**
        maze: *list, list, int*
            A list of lists, holding integers specifying the different aspects
            of the maze:
                0 - Black - A wall
                1 - White - A space to travel in the maze
        pos: *tuple, int*
            The current position of the pointer.
    '''

    def __init__(self, maze, pos=(0, 0)):
        # Save maze as a NumPy array
        self.maze = np.array(maze)
        # Define the target position
        nrows, ncols = self.maze.shape
        self.target = (nrows - 1, ncols - 1)
        # Define valid paths
        self.paths = [(r, c) for r in range(nrows)
                      for c in range(ncols) if self.maze[r, c] == 1]
        # Remove the target position from the list of valid paths
        self.paths.remove(self.target)
        # Checks
        if self.maze[self.target] == 0:
            raise Exception("Invalid Maze: target cell cannot be blocked!")
        if pos not in self.paths:
            raise Exception("Invalid Location")
        # Call reset function
        self.reset(pos)

    def reset(self, pos):
        '''
        Reset the pointer to its starting position
        '''
        self.pos = pos
        self.maze = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        row, col = pos
        # Set maze position
        self.maze[row, col] = 0.5
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        '''
        Update the state of the maze
        '''
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = pos_row, pos_col, mode = self.state
        # Mark a visited space by adding it to the appropriate set
        if self.maze[pos_row, pos_col] > 0:
            self.visited.add((pos_row, pos_col))
        # Obtain list of valid actions
        valid_actions = self.valid_actions()
        # Check if there are any valid actions and adjust position accordingly
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        # If we reach this point, there is an invalid action and we keep the
        # same position
        else:
            nmode = 'invalid'
        # Define the new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        '''
        Get the reward value
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
        Function to call other functions
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
        Obtain the current environment
        '''
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        '''
        Draw the maze environment
        '''
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # Clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0:
                    canvas[r, c] = 1
        # Draw the current pointer
        row, col, valid = self.state
        canvas[row, col] = 0.5
        return canvas

    def game_status(self):
        '''
        Check the status of the game
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
        Get the valid actions
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
        if row > 0 and self.maze[row - 1, col] == 0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0:
            actions.remove(3)
        # Remove LEFT or RIGHT
        if col > 0 and self.maze[row, col - 1] == 0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0:
            actions.remove(2)
        return actions


def show(qmaze):
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
    pos_row, pos_col, _ = qmaze.state
    canvas[pos_row, pos_col] = 0.3   # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.paths:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell
        # (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size,
                                               replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


def qtrain(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []   # history of win/lose game
    n_paths = len(qmaze.paths)
    hsize = qmaze.maze.size // 2   # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmaze.paths)
        qmaze.reset(rat_cell)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()

        n_episodes = 0
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

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss,
                              n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9:
            epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" %
          (epoch, max_memory, data_size, t))
    return seconds


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8     # The current rat cell will be painteg by gray 0.5
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    # Actions dictionary
    actions_dict = {
        LEFT: 'left',
        UP: 'up',
        RIGHT: 'right',
        DOWN: 'down',
    }

    num_actions = len(actions_dict)

    # Exploration factor
    epsilon = 0.1

    maze = np.array([
        [1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [1.,  1.,  1.,  0.,  0.,  1.,  0.],
        [0.,  0.,  0.,  1.,  1.,  1.,  0.],
        [1.,  1.,  1.,  1.,  0.,  0.,  1.],
        [1.,  0.,  0.,  0.,  1.,  1.,  1.],
        [1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [1.,  1.,  1.,  0.,  1.,  1.,  1.]
    ])

    qmaze = Qmaze(maze)
    model = build_model(maze)
    qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)
