'''
Read training and testing data
'''
import struct
import numpy as np


def read_data(fptr):
    with open(fptr, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
    return data


if __name__ == '__main__':
    fptr = "Data/train-images-idx3-ubyte"
    data = read_data(fptr)
    print(type(data))
