# Monte Carlo Simulation

## Project Overview:
In this project, Monte Carlo methods has been used to estimate the value of Pi. Using the concepts of multi-threading in combination with texture memory and fast-math computations.

## Theory:
Value of pi can be obtained as the ratio of area of circle to the area of square in which the circle is inscribed. i.e: `pi*r*r / 4*r*r = pi/4`
Where the area is the number of points present inside the shape, So the accuracy of pi actually depends upon the precise calculation of number of points inside the shape.

`pi = ( 4 * number of points inside cirlce ) / number of points inside the square`

keeping the domain within (0, 1] to avoid the variables exceed the memory limits.
Using a single thread we can generate `10^7` points in a loop within the domain and check whether it lies inside the circle `x*x + y*y = 1` 
In order to achieve more precision and accuracy of pi, we can use multi-threading. Where we will be computing the values of points inside the circle across multiple threads invoking the same kernel, hence for faster memory access by caching the variables being reused texture memory can be employed.


## Requirements:
Cuda Toolkit 12.8+
Nvidia Nsight Visual Studio