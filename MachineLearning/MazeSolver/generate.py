'''
Maze Generation and Solving

Here we will generate and solve mazes. Mazes are generated as png images
using the Python Imaging Library (PIL), and similarly solved by reading in
a png image, and outputing a new one with the solution marked out in green.

*Note*************************************************************************
All coordinates of our maze are initialized as "0" to signify "Black". As
a new coordinate enters the positions stack, it will be come "1" to signify
"White".
******************************************************************************

*METHOD***********************************************************************
Visualization:
    https://en.wikipedia.org/wiki/File:Depth-First_Search_Animation.ogv

The approach taken in this lab is to use a "Depth First Search" approach at
maze generation and solving. This method starts with a "stack", let's call
this stack "positions". It will be initialized at some coordinate to signify
the start of the maze. As such, we may begin at:

    positions = [
        (0, 0)
    ]

We then will look for a valid position to take, and randomly select it.  Two
possibilities now exist:

    1. We take a random step.
    2. No valid options exist.

If (1), then we simply append the new coordinate to the positions stack,
adjust variables accordingly, and continue. If (2), then we "pop" the last
position out of the stack, and continue from where we previously were (seeking
out a new step to take). Note, this backtracking is represented in the
wikipedia example as the maze changing to blue.

Once the entire space of possible choices has been explored, it should be
evident that the backtracking will continue until the positions stack is
empty. It is at this point that the maze generation should end.

When solving, the same approach can be taken. Care needs to be made in
regards to assessing if a step is "valid" or not. Further, a new end criteria
needs making.
******************************************************************************
'''
import random
from PIL import Image
import os


def get_colors():
    '''
    Colors map that the maze will use:
        0 - Black - A wall
        1 - White - A space to travel in the maze
        2 - Green - A valid solution of the maze
        3 - Red - A backtracked position during maze solving
        4 - Blue - Start and Endpoints of the maze
    **Returns**
        color_map: *dict, int, tuple*
            A dictionary that will correlate the integer key to
            a color.
    '''
    return {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (0, 255, 0),
        3: (255, 0, 0),
        4: (0, 0, 255),
    }


def save_maze(maze, blockSize, name, directory=os.getcwd()):
    '''
    This will save a maze object to a file.
    **Parameters**
        maze: *list, list, int*
            A list of lists, holding integers specifying the different aspects
            of the maze:
                0 - Black - A wall
                1 - White - A space to travel in the maze
                2 - Green - A valid solution of the maze
                3 - Red - A backtracked position during maze solving
                4 - Blue - Start and Endpoints of the maze
        blockSize: *int, optional*
            How many pixels each block is comprised of.
        name: *str, optional*
            The name of the maze.png file to save.
        directory: *string*
            The location where the output file will be saved.
    **Returns**
        None
    '''
    nBlocks = len(maze)
    dims = nBlocks * blockSize
    colors = get_colors()
    # Verify that all values in the maze are valid colors.
    ERR_MSG = "Error, invalid maze value found!"
    assert all([x in colors.keys() for row in maze for x in row]), ERR_MSG
    img = Image.new("RGB", (dims, dims), color=0)
    # Parse "maze" into pixels
    for jx in range(nBlocks):
        for jy in range(nBlocks):
            x = jx * blockSize
            y = jy * blockSize
            for i in range(blockSize):
                for j in range(blockSize):
                    img.putpixel((x + i, y + j), colors[maze[jx][jy]])
    if not name.endswith(".png"):
        name += ".png"
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Save the image in the proper directory
    output_string = directory + "/" + name
    img.save(output_string)


def load_maze(filename, blockSize=10, directory=os.getcwd()):
    '''
    This will read a maze from a png file into a 2d list with values
    corresponding to the known color dictionary.
    **Parameters**
        filename: *str*
            The name of the maze.png file to load.
        blockSize: *int, optional*
            How many pixels each block is comprised of.
        directory: *string*
            The location where the output file will be saved.
    **Returns**
        maze: *list, list, int*
            A 2D array holding integers specifying each block's color.
    '''
    if ".png" in filename:
        filename = filename.split(".png")[0]
    img = Image.open(directory + "/" + filename + ".png")
    dims, _ = img.size
    nBlocks = int(dims / blockSize)
    colors = get_colors()
    color_map = {v: k for k, v in colors.items()}
    maze = [[0 for x in range(nBlocks)] for y in range(nBlocks)]
    for i, x in enumerate(range(0, dims, dims // nBlocks)):
        for j, y in enumerate(range(0, dims, dims // nBlocks)):
            px = x
            py = y
            maze[i][j] = color_map[img.getpixel((px, py))]
    return maze


def pos_chk(x, y, nBlocks):
    '''
    Validate if the coordinates specified (x and y) are within the maze.
    **Parameters**
        x: *int*
            An x coordinate to check if it resides within the maze.
        y: *int*
            A y coordinate to check if it resides within the maze.
        nBlocks: *int*
            How many blocks wide the maze is.  Should be equivalent to
            the length of the maze (ie. len(maze)).
    **Returns**
        valid: *bool*
            Whether the coordiantes are valid (True) or not (False).
    '''
    return x >= 0 and x < nBlocks and y >= 0 and y < nBlocks


def generate_maze(nBlocks, name, start, blockSize, slow,
                  directory=os.getcwd()):
    '''
    Generate a maze using the Depth First Search method.
    **Parameters**
        nBlocks: *int*
            The number of blocks in the maze (x and y dims are the same).
        name: *str, optional*
            The name of the output maze.png file.
        start: *tuple, int, optional*
            Where the maze will start from, and the initial direction.
        blockSize: *int, optional*
            How many pixels each block will be.
        slow: *bool, optional*
            Whether to save and lag on generation so as to view the mazegen.
        directory: *string*
            The location where the output file will be saved.
    **Returns**
        None
    '''
    # Initialize maze as an array of black cells
    maze = [[0 for i in range(nBlocks)] for j in range(nBlocks)]
    # Initialize stack of positions and directions
    positions = [start]
    list_of_directions = [0]
    # Define the four directions in which we can move
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    # Define the starting cell of the maze as white
    maze[start[0]][start[1]] = 1
    # Generate the maze
    while len(positions) > 0:
        # Retrieve the end values of the stacks
        current_x, current_y = positions[-1]
        current_d = list_of_directions[-1]
        # Prevent zigzags when generating the map
        if len(list_of_directions) > 2 and current_d != list_of_directions[-2]:
            d_range = [current_d]
        else:
            d_range = range(len(directions))
        # Initialize empty list for neighboring cells
        neighbors = []
        # Check neighboring cells for possible paths to take
        for i in d_range:
            # Define coordinates for next step
            next_x = current_x + directions[i][0]
            next_y = current_y + directions[i][1]
            # Check if specified coordinates are in the maze and if they
            # represent a wall
            if pos_chk(next_x, next_y, nBlocks) and maze[next_x][next_y] == 0:
                # At this point, the number of occupied neighbors must be 1
                count = 0
                # Check if neighboring cells have adjacent paths and determine
                # the possible paths to take
                for j in range(len(directions)):
                    px = next_x + directions[j][0]
                    py = next_y + directions[j][1]
                    if pos_chk(px, py, nBlocks) and maze[px][py] == 1:
                        count += 1
                if count == 1:
                    neighbors.append(i)
        # If 1 or more neighbors are available, then randomly select one and
        # move to it
        if len(neighbors) > 0:
            current_d = neighbors[random.randint(0, len(neighbors) - 1)]
            current_x += directions[current_d][0]
            current_y += directions[current_d][1]
            maze[current_x][current_y] = 1
            positions.append((current_x, current_y))
            list_of_directions.append(current_d)
        else:
            # If there are no possible neighboring cells to move to, remove
            # the current position from the stack
            positions.pop()
            list_of_directions.pop()
        if slow:
            save_maze(maze, blockSize=blockSize, name=name,
                      directory=directory)
    # Save the generated maze and set start/end points
    save_maze(maze, blockSize=blockSize, name=name, directory=directory)


def solve_maze(filename, start, end, blockSize, slow, directory=os.getcwd()):
    '''
    Solve a maze using the Depth First Search method.
    **Parameters**
        filename: *str*
            The name of the maze.png file to be solved.
        start: *tuple, int, optional*
            Where the maze will start from.
        end: *tuple, int, optional*
            Where the maze will end.
        blockSize: *int, optional*
            How many pixels each block will be.
        slow: *bool, optional*
            Whether to save and lag on generation so as to view the mazegen.
        directory: *string*
            The location where the output file will be saved.
    **Returns**
        None
    '''
    # Remove file extension
    if ".png" in filename:
        filename = filename.split(".png")[0]
    # Load the maze
    maze = load_maze(filename, blockSize=blockSize, directory=directory)
    nBlocks = len(maze)
    # Initialize stack of positions
    positions = [start]
    # Define the four directions in which we can move
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    # Ensure that the start and end points are colored appropriately. If end
    # is not valid, this solver will fail!
    maze[start[0]][start[1]] = 4
    maze[end[0]][end[1]] = 4
    # Solve the maze
    while len(positions) > 0:
        # Retrieve the end values of the stack
        current_x, current_y = positions[-1]
        # Initialize empty list for neighboring cells
        neighbors = []
        # Determine which neighboring cells are possible to move to
        for i in range(len(directions)):
            # Define coordinates for next step
            next_x = current_x + directions[i][0]
            next_y = current_y + directions[i][1]
            # Check if specified coordinates are in the maze and if they
            # represent an available path to take
            if pos_chk(next_x, next_y, nBlocks) and (next_x, next_y) != start:
                if maze[next_x][next_y] == 1 or maze[next_x][next_y] == 4:
                    neighbors.append(i)
        # If 1 or more neighbors are available, then randomly select one and
        # move to it
        if len(neighbors) > 0:
            current_d = neighbors[random.randint(0, len(neighbors) - 1)]
            current_x += directions[current_d][0]
            current_y += directions[current_d][1]
            # If we have reached the end of the maze, break out of the while
            # loop
            if (current_x, current_y) == end:
                break
            # Set path to be green
            maze[current_x][current_y] = 2
            positions.append((current_x, current_y))
        else:
            # If there are no possible neighboring cells to move to, remove
            # the current position from the stack and backtrack
            maze[current_x][current_y] = 3
            positions.pop()
        if slow:
            save_maze(maze, blockSize=blockSize,
                      name="%s_solved.png" % filename, directory=directory)
    # Check if a solution has been found
    if not any([m == 2 for row in maze for m in row]):
        print("NO VALID SOLUTION FOR THE CHOSEN ENDPOINT!")
    # Save the solved maze
    save_maze(maze, blockSize=blockSize, name="%ss.png" % filename,
              directory=directory)


if __name__ == "__main__":
    generate_maze(7, name="maze", start=(0, 0), blockSize=10, slow=False)
