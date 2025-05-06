
"""
Regression.
This is finding the relationship between variables.
In ML and statistical modeling, that relationship is used to predict the outcome of future events.
"""

# Linear - uses the relationship between data points to graw a straight line through all of them.
import matplotlib.pyplot as plt
import numpy
from scipy import stats

# Sample data
x = [5,7,8,7,2,17,2,9,4,11,12,9,6] # age of cars
y = [99,86,87,88,111,86,103,87,94,78,77,85,86] # speed of cars

# Linear regression, this fits the line of best fit through the data.
# slope = how much y changes for every 1 unit change in x.
# intercept = the y-value when x = 0
# r = correlation coefficient, closeness of the data to the line(from -1 to 1)
# std_err = standard error of the slope
slope, intercept, r, p, std_err = stats.linregress(x, y)

# define the linear function
def funct(x):
    return slope * x + intercept # y = mx + b

# apply the linear function to each x value to get predicted y values
exampleModel = list(map(funct, x))
plt.scatter(x, y) # plots original data
plt.plot(x, exampleModel) # plots the regression line
plt.show()

### NB. stats.lingress returns 5 values as a tuple. so you can print slope, r, etc. or similarly you can type :
# result = stats.linregress(x, y) and slope = result[0], intercept = result[1] etc... ###


''' R. This shows relationship. 0 means no relationship whereas 1 and -1 means 100% related. '''

print("Relationship between x and y:", r)

# Predicting future values. We can now use the info we gathered to predict future values.

# let's predict the speed of a 10-year-old car

speed1 = funct(10)
print("Predicted speed of a 10 year old car:", speed1) # You can also read this value from the graph!!



'''
####################################################################################################
Polynomial Regression. This is to be used if your data points will clearly not fit a linear regression (straight line).
####################################################################################################
'''

# Lets say we have a dataset where:
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22] # x = the hour of the day a car passes a tollbooth
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100] # y = the speed of the car going past the tollbooth

plt.scatter(x, y) # create scatter plot with x and y
plt.show()

polymodel = numpy.poly1d(numpy.polyfit(x, y, 3)) # method to create a polynomial model

myline = numpy.linspace(1,22,100) # 100 evenly spaced points from 1-22, creates a smooth line
plt.scatter(x, y)
plt.plot(myline, polymodel(myline))
plt.show()

### Like before, it is important to know the relationship between X and Y. No relationship = bad/no prediction.
### Here we measure with a value called R squared. Sklearn will compute this for you.

from sklearn.metrics import r2_score
print("R-Squared:", r2_score(y, polymodel(x))) # 0-1, 0 meaning no relationship and 1 meaning 100% related

# A score of 0.94 shows a very good relationship so we can use polynomial regression in future predictions

### Example. Predict the speed of a car passing the tollbooth at 17:00:

speed = polymodel(17)
print("Predicted speed of a car passing at 17:00:", speed) # this can also be read from the diagram