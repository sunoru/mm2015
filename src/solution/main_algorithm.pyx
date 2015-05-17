# coding=utf-8
from vehicle import cities, load_demand, load_path, save_data, load_data, oo
from solution.method_2 import method_2

def main(area):
    demand = load_demand(area)
    path = load_path(area)
    city_list = cities[area][1]
    method_2(demand, path, city_list, area, True)
