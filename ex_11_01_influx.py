#https://www.influxdata.com/blog/getting-started-python-influxdb/
import numpy
import time
from influxdb import InfluxDBClient
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
json_body = [
    {
        "measurement": "brushEvents",
        "tags": {
            "user": "Carol",
            "brushId": "6c89f539-71c6-490d-a28d-6c5d84c0ee2f",
            "factory": "1",
            "camera": "4"
        },
        "time": "2018-03-28T8:01:00Z",
        "fields": {
            "granularity_1": 127,
            "granularity_2": 27
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
json_body2 = [
    {
        "measurement": "cameraEvent",
        "time": "2018-03-28T8:01:00Z",
        "fields": {
            "granularity_XS": 0.25,
            "granularity_S": 0.27,
            "granularity_M": 0.6,
            "granularity_L": 0.1,
            "granularity_XL": 0.2,
        },
        "tags": {
            "FactoryName": "Kyiv",
            "camId": "Cam_Kyiv_1",

        },
    }
]
# ----------------------------------------------------------------------------------------------------------------------
class Influx_connector(object):
    def __init__(self,host='localhost', port=8086,database='pyexample'):
        self.database = database
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
    def persist_measurement(self, json_body):
        self.client.write_points(json_body)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def read_measurement(self, query):
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
    def remove_measurement(self,measurement,tags=None):
        self.client.delete_series(database=self.database, measurement=measurement, tags=tags)
        return
# ----------------------------------------------------------------------------------------------------------------------
class TS_generator(object):
    def __init__(self,verbose=False):
        self.measurement_name = 'measurement_QC'
        self.fields = ['granularity_XS','granularity_S','granularity_M','granularity_L','granularity_XL']
        self.avgs   = [10, 20, 20, 30, 40]
        self.devts  = [1, 2, 4, 1, 8]
        self.phs    = [0.2, 0.05, 0.1, 0.01, 0.001]
        self.verbose = verbose
        self.prev = {}
        for each in self.fields:self.prev[each]=int(50)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate_TS(self,duration,cam_mame='Cam_Kyiv_01',location_name='Kyiv'):
        json_body = []
        weight = 1.0

        for t in range(duration):
            time.sleep(0.2)
            seconds = int(time.strftime("%S", time.gmtime()))
            dct_fields = {}
            for field,a,d,ph in zip(self.fields, self.avgs, self.devts,self.phs):
                noise = (numpy.random.rand(1)[0] - 0.5) * d
                trend = a
                seasonality = a*0.2*(numpy.sin(ph*seconds))
                value = trend + seasonality + noise
                value = self.prev[field]*(1-weight) + (weight)*value

                dct_fields[field] = int(numpy.clip(value,0,100))
                self.prev[field] = float(value)

            dct_tags = {}
            dct_tags['location_name'] = location_name
            dct_tags['cam_name'] = cam_mame

            dct = {}
            dct['measurement'] = self.measurement_name
            dct['time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) #"2018-03-30T8:02:00Z"
            dct['fields'] = dct_fields
            dct['tags'] = dct_tags

            json_body.append(dct)

        json_body = self.normalize_TS(json_body)
        if self.verbose:
            print(dct['time'],end=' ')
            print(self.fields[1],'%2.2f'%dct_fields[self.fields[1]])

        return json_body
# ----------------------------------------------------------------------------------------------------------------------
    def normalize_TS(self,json_body):
        for record in json_body:
            S = numpy.sum([record['fields'][c] for c in record['fields']])
            for c in record['fields']:
                record['fields'][c]=100*record['fields'][c]/S

        return json_body
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    C = Influx_connector()
    G = TS_generator(verbose=True)
    C.remove_measurement('measurement_QC')

    while True:
        json_body = G.generate_TS(10)
        C.persist_measurement(json_body)

