#!/usr/bin/env python
__author__ = "Guillaume Bellec"

'''
    File name: ex1_run_one_algorithm.py
    Date created: 8/29/2016
    Date last modified: 10/14/2016
    Python Version: 3.4
    Course: Autonomous Learning Systems (TU Graz)

    This file contains the definition of the cost functions used in the first exercise.
    Nothing to be modified by the student in this file.

'''

import numpy as np

def rastrigin(x):
    x = np.array(x)
    return np.sum(x**2 + 10 - 10* np.cos(2* np.pi*x))
bound_rastrigin = [-5,5]

def rosenbrock(x):
    x = np.array(x)
    return (x[1]-x[0]**2)**2 + (x[0]-1)/2 + 2*np.sum(np.abs(x-1.5))
bound_rosenbrock = [-2,2]

def ackley(x):
    x = np.array(x)
    return np.exp(1) + 20 - 20*np.exp(-0.2*np.sqrt(1/2*np.sum(x**2))) - np.exp(0.5*np.sum(np.cos(2*np.pi*x)))
bound_ackley = [-2,2]

def chasm(x):
    x = np.array(x)
    return 1e3*np.abs(x[0])/(1e3*np.abs(x[0])+1) + 1e-2*np.abs(x[1])
bound_chasm = [-5,5]

def get_cost_function(name):
    '''
    Return the cost functions and bounds of solutions for a given cost function.

    :param name: name of the cost function
    :return: A tuple with first the cost function and second the bound on the format [x_min,x_max] with x_min and x_max being scalars
    '''
    cost_functions = {}

    # List the possible things to pack
    name_list = ['rastrigin','rosenbrock','ackley','chasm']
    function_list = [rastrigin,rosenbrock,ackley,chasm]
    bound_list = [bound_rastrigin,bound_rosenbrock,bound_ackley,bound_chasm]

    # Create a dictionnary which associate the function and state bound to a cost name
    for n,f,bound in zip(name_list,function_list,bound_list):
        cost_functions[n] = {'f': f, 'bound': bound}

    print('Function: {}'.format(name))
    return cost_functions[name]['f'],cost_functions[name]['bound']

def plot_cost(ax,f,bound):
    '''
    Plotting the 3D surface of for a given cost function.
    :param ax: Axis3D object from matplotlib
    :param f: The function to optimize
    :param bound: The bounds of the solution space
    :return:
    '''
    n = 200

    x_ax = np.linspace(bound[0],bound[1],n)
    y_ax = np.linspace(bound[0],bound[1],n)
    XX,YY = np.meshgrid(x_ax,y_ax)

    ZZ = np.zeros(XX.shape)
    for i in range(n):
        for j in range(n):
            ZZ[i,j] = f([XX[i,j],YY[i,j]])

    ax.plot_surface(XX,YY,ZZ, cmap='jet')
