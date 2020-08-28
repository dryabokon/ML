#https://www.influxdata.com/blog/getting-started-python-influxdb/
import numpy
import os
import time
from influxdb import InfluxDBClient
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
class Influx_connector(object):
    def __init__(self,host='localhost', port=8086,database='pyexample'):
        self.client = InfluxDBClient(host, port)
        #client.create_database('pyexample')
        self.client.switch_database(database)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def list_databases(self):
        res = self.client.get_list_database()
        print(res)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def do_write_query(self,json_body):
        xx = self.client.write_points(json_body)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def do_read_query(self,query):
        result = self.client.query(query)

        result = result.raw['series'][0]
        name = result['name']
        cols = result['columns']
        rows = result['values']

        for col in cols:print(col, end='\t')
        print()
        for row in rows:
            for value in row:print(value, end='\t')
            print()
        return
# ----------------------------------------------------------------------------------------------------------------------
json_body = [
    {
        "measurement": "brushEvents",
        "tags": {
            "user": "Carol",
            "brushId": "6c89f539-71c6-490d-a28d-6c5d84c0ee2f"
        },
        "time": "2018-03-28T8:01:00Z",
        "fields": {
            "duration": 127
        }
    },
    {
        "measurement": "brushEvents",
        "tags": {
            "user": "Carol",
            "brushId": "6c89f539-71c6-490d-a28d-6c5d84c0ee2f"
        },
        "time": "2018-03-29T8:04:00Z",
        "fields": {
            "duration": 132
        }
    },
    {
        "measurement": "brushEvents",
        "tags": {
            "user": "Carol",
            "brushId": "6c89f539-71c6-490d-a28d-6c5d84c0ee2f"
        },
        "time": "2018-03-30T8:02:00Z",
        "fields": {
            "duration": 129
        }
    }
]
# ----------------------------------------------------------------------------------------------------------------------
def generate_TS(N,measurement_name='brushEvents',field_name='duration'):
    json_body = []

    for t in range(N):
        time.sleep(0.1)
        dct_fields = {}
        dct_fields[field_name] = int(numpy.random.rand(1)[0]*100)
        dct = {}
        dct['measurement'] = measurement_name
        dct['time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) #"2018-03-30T8:02:00Z"
        dct['fields'] = dct_fields
        json_body.append(dct)

    return json_body
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    C = Influx_connector()
    json_body2 = generate_TS(100)
    C.do_write_query(json_body2)
    C.do_read_query('SELECT "duration" FROM "pyexample"."autogen"."brushEvents" ')
