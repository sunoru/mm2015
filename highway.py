# coding:utf-8
import requests
import json
import codecs

obj_url = 'http://glcx.moc.gov.cn/chinahighway/highwayInformation.do'
headers = {
    'Host': 'glcx.moc.gov.cn',
    'Content-Length': '93',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML,                                                                                                                 like Gecko) Chrome/41.0.2272.118 Safari/537.36',
    'Connection': 'keep-alive',
    'Accept-Encoding': 'gzip, deflate',
    'X-Requested-With': 'XMLHttpRequest',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Origin': 'http://glcx.moc.gov.cn',
    'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6,ja;q=0.4',
    'Referer': 'http://glcx.moc.gov.cn/chinahighway/index.do',
    'Cookie': 'JSESSIONID=B36FAF09CA30EA987700F59FD28C2247; Hm_lvt_1d4daef6965293c83e12d3322f004ad3=1431099549; Hm_lpvt_1d4daef6965293c83e12d3322f004ad3=1431099761',
    'Content-Type': 'application/x-www-form-urlencoded'
}
data = {
    "road_code": "",  # "G2",
    "road_name": "",
    "facility_type": "-1",
    "perPage": "7",
    "page": "1",
    "scale": "8",
    "up_down_mark": "0",
    "road_uuid": ""  # "1001"
}
with codecs.open('highways.txt', encoding='utf-8') as fi:
    highways = [x.split() for x in fi.readlines()]

#highways = [highways[0]]
i = 999
for highway in highways:
    print highway[0],
    print highway[1]
    data["road_code"] = highway[0]
    i += 1
    data["road_uuid"] = str(i)
    x = requests.post(obj_url, headers=headers, data=data)
    x.encoding = 'utf-8'
    with codecs.open('highway_data/%s.txt' % highway[0], 'w', encoding='utf-8')  as fo:
        fo.write(x.text)
