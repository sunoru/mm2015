# coding=utf-8
from deap import creator, base, tools, algorithms
import numpy as np
from vehicle import cities, load_demand, load_path, save_data, load_data


def init_creators():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def do_process(demand_in, demand_out, path_forward, path_back):
    cdef int num = len(demand_in)
    cdef int i
    for i in xrange(num):
        pass


def main(area):
    demand_in = load_demand(area)
    demand_out = demand_in.transpose()
    path_forward = load_path(area)
    assert isinstance(path_forward, np.ndarray)
    path_back = path_forward.transpose()
    city_list = cities[area][1]
    cdef int start = 0
    cdef int total = len(demand_in)
    init_creators()
    while start < total:
        do_process(
            demand_in[start][start:], demand_out[start][start:],
            path_forward[start][start:], path_back[start][start:]
        )
        start += 1
