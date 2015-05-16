# coding=utf-8
from __future__ import division
import os
from deap import creator, base, tools
import random
import sys
from vehicle import cities, oo, save_data, load_data
from solution import algorithms
import numpy as np
cimport numpy as cnp


cdef class Meta(object):
    cdef public cnp.ndarray demand
    cdef public cnp.ndarray path
    cdef public list city_list
    cdef public int num
    cdef public cnp.ndarray priority_scale
    cdef public double speed
    cdef public tuple time_window
    cdef public double time_stay
    cdef public int current_real_num
    cdef public int current_num
    cdef public int start_number
    cdef public int end_number
    cdef public cnp.ndarray current_cities
    cdef public str area
    cpdef double get_distance(self, int x, int y=-1) except *:
        if x >= self.current_num:
            x %= self.current_num
        if y == -1:
            return self.path[self.current_cities[0]][self.current_cities[x]]
        if y >= self.current_num:
            y %= self.current_num
        return self.path[self.current_cities[x]][self.current_cities[y]]
    cpdef cnp.ndarray get_demand(self, tdemand, int x):
        if x > self.current_num:
            x %= self.current_num
        return tdemand[self.current_cities[x]]
    cpdef int compare(self, int x, int y) except *:
        if x >= self.current_num:
            x %= self.current_num
        if y >= self.current_num:
            y %= self.current_num
        if self.current_priority[x] >= self.current_priority[y]:
            return True
        return False
    cdef public cnp.ndarray current_priority
    cdef public object toolbox
    cdef public int MU, LAMBDA, NGEN
    cdef public double CXPB, MUTPB, HEPB, MUTAPB, MUTBPB, MUTCPB
    cdef public str log_dir


cdef Meta meta = Meta()


cdef class Vehicle(object):
    cdef public cnp.ndarray data
    cdef public list extra
    def __init__(self, data, extra=None):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.extra = extra if extra is not None else list()


def errlog(*message):
    for each in message:
        sys.stderr.write("%s\n" % each)


cdef inline double priority_func(int u, double max_dist):
    cdef cnp.ndarray w = meta.priority_scale
    cdef double dtw = meta.time_window[1] - meta.time_window[0]
    cdef double t0 = meta.get_distance(u) / meta.speed
    cdef double result = w[0] * abs(t0 - meta.time_window[0]) / dtw
    result += w[1] * abs(t0 - meta.time_window[1]) / dtw
    result += w[2] * meta.get_distance(u) / max_dist
    return result


def gen_priority(int start):
    cdef int i
    cdef cnp.ndarray priorities
    cdef int q
    priorities = np.zeros(meta.current_num)
    max_dist = np.max(meta.path[start][start:])
    for i in xrange(1, meta.current_num):
        priorities[i] = priority_func(i, max_dist)
    print priorities
    return priorities


def init_individual(u=False):
    cdef int i
    cdef list q = range(1, meta.current_num)
    if u:
        p = np.array((range(meta.current_num), meta.current_priority.copy())).transpose().tolist()
        p.sort(reverse=True, cmp=lambda x, y: -1 if x[1] < y[1] else 1)
        p.remove([0.0, 0.0])
        return creator.Individual(np.array(p, dtype=int).transpose()[0])
    random.shuffle(q)
    return creator.Individual(q)


def init_population(int n=-1, city_name=None):
    cdef int i
    if city_name is not None:
        x = load_data("method_2_population_%s.dat" % city_name)
        if x is not None:
            return [creator.Individual(x[i]) for i in xrange(len(x))]
    if n < 0:
        raise ValueError
    return [meta.toolbox.individual() for i in xrange(n-1)] + [meta.toolbox.individual(True)]


cdef inline double ftime(d):
    return d / meta.speed


cdef inline int check(city, ttime):
    cdef double dist = meta.get_distance(city)
    if dist < 400:
        return meta.time_window[0] < ttime < meta.time_window[1]
    elif dist < 1200:
        return ttime - 24 < meta.time_window[1]
    elif dist < 3000:
        return ttime - 48 < meta.time_window[1]
    else:
        return ttime - 72 < meta.time_window[1]


cdef inline double refresh(ttime):
    cdef int k = 0
    while meta.time_window[1] + k*24 < ttime:
        k += 1
    if ttime < meta.time_window[0] + k*24:
        ttime = meta.time_window[0] + k*24
    return ttime


cdef inline int check_time(cnp.ndarray data):
    cdef int i
    cdef double ttime = ftime(meta.get_distance(data[0]))
    ttime = refresh(ttime)
    if not check(data[0], ttime):
        return False
    for i in xrange(1, len(data)):
        ttime += 2 + ftime(meta.get_distance(data[i-1], data[i]))
        ttime = refresh(ttime)
        if not check(data[i], ttime):
            return False
    return True


cdef inline double calc_dist(cnp.ndarray data, cnp.ndarray tdemand):
    cdef int i
    cdef double tdist = meta.get_distance(data[0])
    cdef list pdemand = []
    cdef double cs = 0.0
    cdef double ct = 0.0
    for i in xrange(len(data)):
        pdemand.append(meta.get_demand(tdemand, data[i]))
    for x in pdemand:
        cs += x[0]
    cs =  min(1.0, cs)
    dt = min(1, pdemand[0][0])
    pdemand[0][0] -= dt
    cs -= dt
    dt = min(1 - cs - ct, pdemand[0][1])
    pdemand[0][1] -= dt
    ct += dt
    for i in xrange(1, len(data)):
        tdist += meta.get_distance(data[i-1], data[i])
        if pdemand[i][0] == pdemand[i][1] == 0 or cs == 0 and ct == 1:
            return -1
        dt = min(1, pdemand[i][0])
        pdemand[i][0] -= dt
        cs -= dt
        dt = min(1 - cs - ct, pdemand[i][1])
        pdemand[i][1] -= dt
        ct += dt
    return tdist + meta.get_distance(data[-1], meta.current_cities[0])


def eval_path(individual):
    cdef int i
    cdef list split_list = [0]
    cdef int l = len(individual.data)
    cdef float pdist, tdist
    for i in xrange(1, l):
        if meta.compare(individual.data[i-1], individual.data[i]):
            split_list.append(i)
    split_list.append(l)
    for i in xrange(1, len(split_list)):
        if not check_time(individual.data[split_list[i-1]:split_list[i]]):
            return oo,
    tdist = 0
    tdemand = np.array((meta.demand[meta.current_cities[0], :], meta.demand[:, meta.current_cities[0]])).transpose()
    for i in xrange(1, len(split_list)):
        pdist = calc_dist(individual.data[split_list[i-1]:split_list[i]], tdemand)
        if pdist == -1:
            return oo,
        tdist += pdist
    for i in xrange(meta.current_cities[0]+1, meta.num):
        if not tdemand[i][0] == tdemand[i][1] == 0:
            return oo,
    return tdist,


def index(ind, z, q):
    nj = np.nonzero(ind[z:]==q)[0]
    if len(nj):
        return nj[0] + z
    else:
        return -1


def rotate(ind, z, p, l):
    if z == l:
        return z
    if p == -1:
        return z
    if z == p:
        return z + 1
    if p == l-1:
        ind[z], ind[p] = ind[p], ind[z]
        ind[z+1:p] = ind[p-1:z:-1]
        return z+1
    if meta.compare(ind[p-1], ind[p+1]):
        ind[z:l-p+z], ind[l-p+z:] = ind[p:].copy(), ind[z:p].copy()
    else:
        ind[z], ind[p] = ind[p], ind[z]
        ind[z+1:p], ind[p+1:] = ind[p-1:z:-1].copy(), ind[:p:-1].copy()
    return z + 1


def mate_mx3(individual1, individual2):
    cdef cnp.ndarray ind1, ind2
    ind1 = individual1.data
    ind2 = individual2.data
    l1, l2 = len(ind1), len(ind2)
    if l1 > l2:
        l1, l2 = l2, l1
        individual1, individual2 = individual2, individual1
        ind1, ind2 = ind2, ind1
    ind3 = {}
    extra = []
    i = random.randrange(l1)
    j = index(ind2, 0, ind1[i])
    ind3[0] = ind1[i]
    if ind3[0] in individual1.extra:
        extra.append(ind3[0])
    zi = rotate(ind1, 0, i, l1)
    zj = rotate(ind2, 0, j, l2)

    for k in xrange(1, l1):
        if meta.compare(ind1[zi], ind2[zj]):
            ind3[k] = ind2[zj]
            if ind3[k] in individual2.extra:
                extra.append(ind3[k])
            i = index(ind1, zi, ind3[k])
            zi = rotate(ind1, zi, i, l1)
            zj += 1
        else:
            ind3[k] = ind1[zj]
            if ind3[k] in individual1.extra:
                extra.append(ind3[k])
            j = index(ind2, zj, ind3[k])
            zi += 1
            zj = rotate(ind2, zj, j, l2)

    p = l1
    for k in xrange(l2 - l1):
        if random.random() < meta.HEPB:
            if zi < l1:
                if meta.compare(ind1[zi], ind2[zj]):
                    ind3[p] = ind2[zj]
                    if ind3[p] in individual2.extra:
                        extra.append(ind3[p])
                    zj += 1
                else:
                    ind3[p] = ind1[zi]
                    if ind3[p] in individual1.extra:
                        extra.append(ind3[p])
                    zi += 1
            else:
                ind3[p] = ind2[zj]
                if ind3[p] in individual2.extra:
                    extra.append(ind3[p])
                zj += 1
            p += 1
    ind3_array = np.zeros(len(ind3))
    for eachkey in ind3:
        ind3_array[eachkey] = ind3[eachkey]
    return creator.Individual(ind3_array, extra), oo


def mutate_method(individual):
    cdef float r = random.random()
    cdef int l = len(individual.data)
    if r < meta.MUTAPB:
        a, b = random.randrange(l), random.randrange(l)
        individual.data[a], individual.data[b] = individual.data[b], individual.data[a]
    elif r < meta.MUTAPB + meta.MUTBPB:
        a = random.randrange(meta.current_real_num)
        p = meta.current_num + a
        while p in individual.extra:
            p += meta.current_num
        individual.data = np.append(individual.data, p)
        individual.extra.append(p)
    else:
        if len(individual.extra):
            p = random.choice(individual.extra)
            pl = individual.data.tolist()
            pl.remove(p)
            individual.extra.remove(p)
            individual.data = np.array(pl)
    return individual,


def clone_individual(individual):
    return creator.Individual(individual.data, individual.extra[:])


def get_fitness(x):
    return x.fitness.values[0]


def select_tournament(individuals, k, tournsize):
    chosen = [min(individuals, key=get_fitness)]
    for i in xrange(k - 1):
        aspirants = tools.selRandom(individuals, tournsize)
        chosen.append(min(aspirants, key=get_fitness))
    return chosen


def init_deap():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", Vehicle, fitness=creator.FitnessMin)
    meta.toolbox = base.Toolbox()
    meta.toolbox.register("clone", clone_individual)
    meta.toolbox.register("attr_bool", random.randint, 0, 1)
    meta.toolbox.register("individual", init_individual)
    meta.toolbox.register("population", init_population)
    meta.toolbox.register("evaluate", eval_path)
    meta.toolbox.register("mate", mate_mx3)
    meta.toolbox.register("mutate", mutate_method)
    meta.toolbox.register("select", select_tournament, tournsize=3) # 可以试试别的


def init_process(int start):
    cdef dict cts = {}
    cdef int i

    meta.current_real_num = meta.current_num = meta.num - start
    for i in xrange(meta.current_num):
        cts[i] = i + start
        for j in xrange(max(int(meta.demand[start][i+start]), int(meta.demand[i+start][start]))):
            cts[meta.current_num] = i + start
            meta.current_num += 1
    meta.current_cities = np.zeros(meta.current_num, dtype=int)
    for i in cts:
        meta.current_cities[i] = cts[i]

    meta.current_priority = gen_priority(start)


def do_process(int start):
    print meta.city_list[start]

    init_process(start)
    pop = meta.toolbox.population(n=meta.MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    tmp = sys.stdout
    with open(os.path.join(meta.log_dir, "%s_%s_%s.log" % (cities[meta.area][0], start, meta.city_list[start])), 'w') as fo:
        sys.stdout = fo
        algorithms.eaMuPlusLambda(
            pop, meta.toolbox, meta.MU, meta.LAMBDA, meta.CXPB, meta.MUTPB, meta.NGEN, stats,
            halloffame=hof, verbose=True
        )
    sys.stdout = tmp

    return pop, stats, hof


def method_2(demand, path, city_list, area):
    meta.demand = demand
    meta.path = path
    meta.city_list = city_list
    meta.area = area
    meta.num = len(demand)
    meta.speed = 70.0
    meta.time_window = (0.0, 12.0)  # 以晚上七点为0，第二天早上七点即为12
    meta.time_stay = 2.0
    meta.MU = 80
    meta.LAMBDA = 160
    meta.CXPB = 0.6
    meta.MUTPB = 0.3
    meta.HEPB = 0.5
    meta.NGEN = 2000
    meta.MUTAPB, meta.MUTBPB, meta.MUTCPB = 0.8, 0.1, 0.1
    meta.priority_scale = np.array([0.2, 0.2, 0.6])
    meta.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log")
    meta.start_number, meta.end_number = 0, meta.num-1
    while True:
        command = raw_input(
            'These parameters will be used:\nspeed: %s\ntime_window: %s\ntime_stay: %s\nMU: %s\n'
            'LAMBDA: %s\nCXPB: %s\nMUTPB: %s\nHEPB: %s\nMUTAPB: %s\nMUTBPB: %s\nMUTCPB: %s\n'
            'NGEN: %s\npriority scale: %s\nlog dir: %s\nstart number: %s\nend number: %s\nInput "OK" to start...\n' % (
                meta.speed, meta.time_window, meta.time_stay, meta.MU, meta.LAMBDA,
                meta.CXPB, meta.MUTPB, meta.HEPB, meta.MUTAPB, meta.MUTBPB, meta.MUTCPB,
                meta.NGEN, meta.priority_scale, meta.log_dir, meta.start_number, meta.end_number
            )
        )
        if command == "OK":
            break
        try:
            cd = command.split(':')
            setattr(meta, cd[0], eval(cd[1]))
            print("%s has been done!\n" % command)
        except:
            continue

    init_deap()
    cdef int i
    for i in xrange(meta.start_number, meta.end_number):
        pop, stats, hof = do_process(i)
        with open(os.path.join(meta.log_dir, 'result.log'), 'a') as fo:
            fo.write("%s\n%s\n%s\n" % (meta.city_list[i], hof[0].data, hof[0].fitness.values[0]))
