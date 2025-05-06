"""
Moving on to work with larger datasets.
Using tools such as distributions, Scatter Plots, Linear/Polynomial/Multiple Regression.
"""

import numpy as np
import numpy.random

# Here we will use a larger uniform data set from 0.0 - 5.0 consisting of 250 random floats
x = numpy.random.uniform(0.0, 5.0, 250)
print(x)

"""
To visualise this data we can draw a histogram. To do so we will use Matplotlib
"""
import matplotlib.pyplot as plt

plt.hist(x, 5) # groups data into 5 bins. This will show us how many values are between 0-1, 1-2, 2-3, 3-4 and 4-5.
plt.show() # displays the graph

"""
Now we will visualise data with a normal (Known as Gaussian) distribution.
"""
x = numpy.random.normal(5.0, 1.0, 100000) # mean of 5, std of 1.0 and 100000 entities.
plt.hist(x, 100)
plt.show()

"""
Scatter plot. This is a graph which represents each datapoint in a dataset with a dot.
"""
# Two data sets of the same length are needed to draw.
x = [5,7,8,7,2,17,2,9,4,11,12,9,6] # x can be the age of a car
y = [99,86,87,88,111,86,103,87,94,78,77,85,86] # y can represent the speed of each car.

plt.scatter(x, y)
plt.show() # what we can see from this graph is the 2 fastest cars are 2 years old and the slowest is 12 years old.
# This infers that newer is faster. This 'could' be a coincidence as we only registered 13 cars.
