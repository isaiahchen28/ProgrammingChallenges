'''
Maze Solver

Colors map that the maze will use:
        0 - Black - A wall
        1 - White - A space to travel in the maze
        2 - Green - A valid solution of the maze
        3 - Red - A backtracked position during maze solving
        4 - Blue - Start and Endpoints of the maze
'''
import numpy as np
import MazeGeneration


def convert_to_numpy(maze):
    '''
    This function will convert a maze to a NumPy array.
    **Parameters**
        maze: *list, list, int*
            A 2D array holding integers specifying each block's color.
    **Returns**
        maze_numpy:
    '''
    return np.array(maze)


def get_directions():
    '''
    Map for 4 directions:
        0 - Left
        1 - Up
        2 - Right
        3 - Down
    **Returns**
        direction_map: *dict, int, str*
            A dictionary that will correlate the integer key to
            a direction.
    '''
    return {
        0: "left",
        1: "up",
        2: "right",
        3: "down",
    }


def main():
    # Get color dictionary and actions dictionary
    colors = MazeGeneration.get_colors()
    actions = get_directions()
    # Get the number of possible actions
    num_actions = len(actions)
    # Define exploration factor
    eps = 0.1
    # Test
    maze = MazeGeneration.load_maze("training/0.png", )
    print(type(maze))




if __name__ == '__main__':
    main()
