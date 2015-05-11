# coding:utf-8
import os
import json
import cPickle
import sys
import re
import numpy as np

import requests

from bases import cities


def save_data(filename, data):
    with open(os.path.join("data", filename), 'wb') as fo:
        cPickle.dump(data, fo)


def load_data(filename):
    if not os.path.exists(os.path.join("data", filename)):
        return None
    with open(os.path.join("data", filename), 'rb') as fi:
        return cPickle.load(fi)


def get_geo(city):
    print "Downloading...",
    obj_url = "http://glcx.moc.gov.cn/chinahighway/poiInfoList.do"
    data = {
        "city": "",
        "keyword": city,
        "searchType": "name"
    }
    headers = {
        'Host': 'glcx.moc.gov.cn',
        'Content-Length': '57',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.118 Safari/537.36',
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate',
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Origin': 'http://glcx.moc.gov.cn',
        'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6,ja;q=0.4',
        'Referer': 'http://glcx.moc.gov.cn/chinahighway/index.do',
        'Cookie': 'JSESSIONID=330858C1635DC4487DF63E90D9AB22BE; '
                  'Hm_lvt_1d4daef6965293c83e12d3322f004ad3=1431325902; '
                  'Hm_lpvt_1d4daef6965293c83e12d3322f004ad3=1431331628',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    x = requests.post(obj_url, headers=headers, data=data)
    x.encoding = 'utf-8'
    response = json.loads(x.text)
    geo = response["pois"]["poi"][0]["geo"]["value"][6:-1].split()
    return geo


def init_geo_data(area):
    data_file = "geo_data_%s.dat" % area
    print "Reading Data..."
    geo_data = load_data(data_file)
    if geo_data:
        return geo_data
    else:
        geo_data = {"name": area, "area": cities[area][0], "data": []}
    for i in xrange(len(cities[area][1])):
        city = cities[area][1][i]
        print city + "...",
        geo_data["data"].append((city, get_geo(city)))
        print "Done!"
    save_data(data_file, geo_data)

    return geo_data


_length_regex = re.compile(r'"length":([\d\.e\+]+)')


def get_distance(x, y):
    obj_url = "http://219.143.235.67:5000/navigation_car?"
    params = "zoom=10&format=json&enctype=pl&" \
             "loc=%s,%s&loc=%s,%s&" \
             "drag=false&etabase=nt&etatime=&rescond=&navtype=fastest&capacity=1.6&" \
             "secc=false&dtirc=false&jsonp=PALM.JSONP.callbacks.route" % (
        x[1][1], x[1][0], y[1][1], y[1][0]
    )
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, sdch",
        "Accept-Language": "zh-CN,zh;q=0.8,en;q=0.6,ja;q=0.4",
        "Connection": "keep-alive",
        "Host": "219.143.235.67:5000",
        "Referer": "http://glcx.moc.gov.cn/chinahighway/index.do",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.118 Safari/537.36"
    }
    q = requests.get(obj_url, params, headers=headers)
    distance = float(_length_regex.findall(q.text)[0]) / 1000
    return distance


def init_path_data(geo_data):
    data_file = "distance_data_%s.dat" % geo_data["name"]
    path_data = load_data(data_file)
    data = geo_data["data"]
    l = len(data)
    if path_data is None:
        path_data = np.zeros((l, l))
    for i in xrange(l):
        for j in xrange(l):
            if i == j:
                continue
            if path_data[i][j] != 0:
                continue
            print "%s %s" % (data[i][0], data[j][0]),
            try:
                path_data[i][j] = get_distance(data[i], data[j])
            except IndexError:
                path_data[i][j] = -1
            except requests.exceptions.ConnectionError as e:
                save_data(data_file, path_data)
                raise e
            print "%s" % path_data[i][j]
    save_data(data_file, path_data)


def main():
    for each in cities:
        geo_data = init_geo_data(each)
        init_path_data(geo_data)


if __name__ == "__main__":
    main()
