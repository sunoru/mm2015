# coding=utf-8
import codecs
import os
from vehicle.bases import cities, data_dir
from vehicle.data_process import init_geo_data, init_path_data, save_data
import numpy as np


def export_data(area):
    data_file = os.path.join(data_dir, "%s.CSV" % area)
    geo_data = init_geo_data(area)
    path_data = init_path_data(geo_data)
    print geo_data['area'],
    data = geo_data['data']
    l = len(data)
    with codecs.open(data_file, 'w', encoding='utf-8') as fo:
        for i in xrange(l):
            fo.write(",%s" % data[i][0])
        fo.write('\n')
        for i in xrange(l):
            fo.write("%s" % data[i][0])
            for j in xrange(l):
                fo.write(",%s" % path_data[i][j])
            fo.write('\n')
    print "Done!"


def import_data(area):
    print cities[area][0], '...'

    from_file = os.path.join(data_dir, "%s.CSV" % area)
    data_file = "distance_data_%s.dat" % area
    city_list = cities[area][1]
    l = len(city_list)
    s = np.zeros((l, l))
    with codecs.open(from_file, 'r', encoding='utf-8') as fi:
        from_data = fi.read().replace('\r', '').split('\n')
    if city_list != from_data[0].split(',')[1:-1]:
        print city_list
        print from_data[0].split(',')
        return None

    for i in xrange(l):
        t = from_data[i+1].split(',')
        for j in xrange(l):
            s[i][j] = t[j+1]
        print s[i]
    save_data(data_file, s)
    print "Done!"


def main():
    for area in cities:
        #export_data(area)
        import_data(area)

if __name__ == "__main__":
    main()
