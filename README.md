
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

  

# Run:

Compiler used is nvcc

` nvcc --version`

nvcc: NVIDIA (R) Cuda compiler driver

Copyright (c) 2005-2024 NVIDIA Corporation

Built on Wed_Aug_14_10:14:07_PDT_2024

Cuda compilation tools, release 12.6, V12.6.68

Build cuda_12.6.r12.6/compiler.34714021_0

` nvcc pr_pi_rng_mc_fix.cu -o pr_pi_rng_mc_fix -arch=sm_87`
` ./pr_pi_rng_mc_fix`

sample output:

Inside circle: 53972125083, Outside circle: 14747351653

Estimated Pi: 3.14159