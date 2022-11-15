# gradient_descent

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(start, gradient, learn_rate, max_iter, tol=0.001, step_size = 1):
    steps = [start] # history tracking
    cur_x = start
    iters = 0

    while step_size > tol and iters < max_iter:
        prev_x = cur_x #Store current x value in prev_x
        gradient_subtraction = learn_rate * gradient(prev_x)
        cur_x = cur_x - gradient_subtraction #Grad descent
        step_size = abs(cur_x - prev_x) #Change in x
        iters += 1 #iteration count
        steps.append(cur_x)
        print("Iteration",iters,"\nX value is",cur_x)
        print(f"Step_size: {step_size}")
    print(f"The local minimum occurs at {cur_x}")
    return steps, cur_x

function = lambda x: (x + 3)**2
history, result = gradient_descent(3, function, 0.1, 100)

x_ = np.linspace(-7,5,100)
y = function(x_)
# setting the axes at the centre
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# plot the function
plt.plot(x_,y, 'r')
plt.plot(history, function(np.array(history)), '-o')
# show the plot
plt.show()
