# coding=utf-8
from deap import creator, base, tools, algorithms
from vehicle import cities, oo
import numpy as np


def init_creators():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def in_passing(u, float depth, float weight, demand, path,
               list visitted, float max_length, tuple the_result):
    cdef int num = len(demand)
    cdef int i
    for i in xrange(1, num):
        if u == i:
            continue
        if i in visitted:
            continue
        p = demand[i] % 1.0
        if p == 0:
            continue
        if p + weight > 1:
            continue
        if depth + path[u][i] < max_length + path[0][i]:
            # 由于三角形规则当depth=0时上述不等式几乎一定成立
            visitted.append(i)
            in_passing(i, depth + path[u][i], weight + p, demand, path, visitted, max_length, the_result)
            visitted.pop()
    if depth < the_result[0]:
        the_result = (depth, visitted[:])



def do_process(result_list, demand_in, demand_out, path):
    cdef int num = len(demand_in)
    cdef int i, j
    cdef double q = 0.0
    for i in xrange(1, num):
        while demand_in[i] >= 1:

            result_list.append()
            q += path[0, i] + path[i, 0]
            demand_in[i] -= 1
            demand_out[i] = max(demand_out[i] - 1, 0)
        while demand_out[i] >= 1:
            pass

def method_1(demand_in, demand_out, path, city_list):
    cdef int start = 0
    cdef int total = len(demand_in)
    init_creators()
    result_list = []
    while start < total:
        print "start:", start
        do_process(
            result_list,
            demand_in[start][start:], demand_out[start][start:],
            path[start:, start:]
        )
        start += 1