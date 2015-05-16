# coding=utf-8
import codecs
import os
import numpy as np

from vehicle.bases import cities
from vehicle.data_process import load_data, save_data


def load_gdp_pop(area):
    data_file = os.path.join("..", "data", "GDP_pop.txt")
    out_file = "%s_GDP_pop.dat" % area
    odata = load_data(out_file)
    if odata:
        return odata
    city_list = cities[area][1]
    a = {}
    with codecs.open(data_file, 'r', encoding='utf-8') as fi:
        indata = fi.readlines()
    for city in city_list:
        print city + '...',
        o = False
        for q in indata:
            if q.find(city) >= 0:
                a[city] = map(float, q.split()[3:5])
                o = True
                print a[city]
                break
        if not o:
            print "Input: ",
            a[city] = input()
    save_data(out_file, a)
    return a


def load_demand(area, gdp_pops=None):
    data_file = "%s_demand.dat" % area
    odata = load_data(data_file)
    if odata is not None:
        return odata
    if gdp_pops is None:
        raise Exception("No GDP and population data!")
    city_list = cities[area][1]
    l = len(city_list)
    tmap = np.zeros((l, l))
    ps = gdp_pops[u'杭州'][0] * gdp_pops[u'南京'][1]
    for i in xrange(l):
        for j in xrange(l):
            if i == j:
                continue
            x = gdp_pops[city_list[i]][0]
            y = gdp_pops[city_list[j]][1]
            tmap[i][j] = x * y / ps
    save_data(data_file, tmap)
    return tmap


def main():
    for area in cities:
        gdp_pops = load_gdp_pop(area)
        load_demand(area, gdp_pops)

if __name__ == "__main__":
    main()
