# coding:utf-8
import codecs
import os


def main():
    from vehicle.bases import cities
    from vehicle.data_process import init_geo_data, init_path_data
    for area in cities:
        data_file = os.path.join("data", "%s.CSV" % area)
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

if __name__ == "__main__":
    main()
