"""
Project Euler - Problem 3

The prime factors of 13195 are 5, 7, 13 and 29. What is the largest prime
factor of the number 600851475143?
"""
import time


def largest_prime_factor(n):
    i = 2
    while i * i < n:
        while n % i == 0:
            n = n / i
        i += 1
    return n


t0 = time.clock()
print(largest_prime_factor(600851475143))
t1 = time.clock()
print("{0} ms".format((t1 - t0) * 1000))
