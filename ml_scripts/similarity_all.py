# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:40:18 2021

@author: sosi
"""
#Output all the similarity measures that were covered in this course
#SMC, Jaccard, Extended Jaccard, Cosine
#Tested with lecture 3 Quiz 1
#Slide 16 has all the formulas

import numpy as np

### SIMILIARITY MEASURES
def sim(x, y):
    f11 = sum((x == 1) & (y == 1))
    f10 = sum((x == 1) & (y == 0))
    f01 = sum((x == 0) & (y == 1))
    f00 = sum((x == 0) & (y == 0))
    return f11, f10, f01, f00


def SMC(x, y):
    f11, f10, f01, f00 = sim(x, y)
    M = len(x)
    return (f11 + f00) / M


def J(x, y):
    f11, f10, f01, f00 = sim(x, y)
    return f11 / (f11 + f10 + f01)


def cos(x, y):
    f11, f10, f01, f00 = sim(x, y)
    return x.T@y / (np.linalg.norm(x) * np.linalg.norm(y))


def EJ(x, y):
    a = x.T @ y 
    b = np.linalg.norm(x) ** 2 + np.linalg.norm(y) ** 2 - a
    return a / b

def similarity_all(x,y):
    print("SMC:", SMC(x,y))
    print("Jaccard:", J(x,y))
    print("EJ:", EJ(x,y))
    print("Cosine:", cos(x,y))

#Observations O1, O2. What are all the similiarty measures between these two?
x = np.array([1,0,1,0,1,0])
y = np.array([1,0,1,0,0,1])

similarity_all(x,y)
