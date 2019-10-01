"""
Weekly Programmer for the week of January 28

Given n non-negative integers a1, a2, ..., an, where each represents a point at
coordinate (i,ai), n vertical lines are drawn such that the two endpoints of
line i are at (i,ai) and (i,0). Find two lines, which together with the x-axis
forms a container, such that the container contains the most water.

You may not slant the container and n is at least 2.
"""
import time


def max_area_pointers(heights):
    area = 0
    p1 = 0
    p2 = len(heights) - 1
    while (p1 < p2):
        area = max(area, min(heights[p1], heights[p2]) * (p2 - p1))
        if (heights[p1] < heights[p2]):
            p1 += 1
        else:
            p2 -= 1
    return area


t0 = time.clock()
print(max_area_pointers([1, 8, 6, 2, 5, 4, 8, 3, 7]))
t1 = time.clock()
print("{0} ms".format((t1 - t0) * 1000))
