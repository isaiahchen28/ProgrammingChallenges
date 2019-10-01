"""
Project Euler - Problem 1

If we list all the natural numbers below 10 that are multiples of 3 or 5, we
get 3, 5, 6 and 9. The sum of these multiples is 23. Find the sum of all the
multiples of 3 or 5 below 1000.
"""


def sum_of_multiples(i, j, n):
    total = 0
    for k in range(n):
        if (k % i == 0 or k % j == 0):
            total += k
    return total


print(sum_of_multiples(3, 5, 1000))
