# Find the greatest common divisor of 2 numbers.
import random
import time


def gcd(x, y):
    if y == 0:
        return x
    else:
        return gcd(y, x % y)


a = random.randint(50, 100)
b = random.randint(50, 100)
t0 = time.clock()
print("GCD of {0} and {1} is {2}.".format(a, b, gcd(a, b)))
t1 = time.clock()
print("{0} ms".format((t1 - t0) * 1000))
