# Covering basic maths for beginners to ML

"""
The most basic 'stats' we all come across are Mean/ Median and Mode,
which are basic level values that are of interest to us in Machine Learning
"""
import numpy as np # used for numerical operations
from scipy import stats

# Speed of cars. The sample data we will be learning with.
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Mean (average) speed
mean_speed = np.mean(speed)
print("Mean speed:", mean_speed)

# Median (middle) speed
median_speed = np.median(speed)
print("Median speed:", median_speed)

# Mode (Most common value) speed. Numpy does not currently contain mode. Use stats library
mode_speed = stats.mode(speed)
print("Mode of speed:", mode_speed) # returns an object which contains both mode and count

"""
Standard deviation. This is a measure of how spread out the data is.
A higher std means values are spread out over a higher range.
"""
x = np.std(speed)
print("Standard deviation:", x) # this means that most of the values are within the range of x from the mean 89.77


"""
Variance is another measure of how spread out the data is.
the sqrt of the Variance is the std! 
"""
x = np.var(speed) # for each value in arr, subtract the mean, square each value and then find the avg of those squared differences
print("Variance:", x)

"""
Percentiles.
These are used to give you a number which describes the value, that a given percentage of the values are lower than.
"""
# ages of people dataset
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
# What is the 75th percentile? meaning what is the age that 75% of people are younger than.
x = np.percentile(ages, 75)
print("Percentiles:", x) # this means 75% of people are 43 y/o or younger.

