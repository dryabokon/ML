import pandas as pd
import cv2
import os
import numpy
import datetime
# ----------------------------------------------------------------------------------------------------------------------
import tools_GIS
import tools_IO
import tools_draw_numpy
import tools_time_convertor
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_GIS/'
folder_out = './data/output/'
filename_in = folder_in + 'solidario_filtered.csv'
# ----------------------------------------------------------------------------------------------------------------------
G = tools_GIS.tools_GIS(folder_out)
import folium
from folium.plugins import TimestampedGeoJson
# ----------------------------------------------------------------------------------------------------------------------
def ex_folium_simple():
    gis_points = numpy.array([
        # (41.8781, -87.6298), #Chicago
        (40.7589, -73.9851),  # Times Square
        (40.7488, -73.985428),  # Empire State Building
        (40.7128, -74.0060),  # New York City
    ])

    G.build_folium_html(gis_points, 'NY_folium.html')
    G.html_to_png(os.getcwd().replace('\\','/')+folder_out[1:]+'NY_folium.html','NY_folium.png',W=300,H=600)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_folium_complex():
    gis_points = numpy.array([
        #(41.8781, -87.6298), #Chicago
        (40.7589, -73.9851),  # Times Square
        (40.7488, -73.985428),  # Empire State Building
        (40.7128, -74.0060),  # New York City
    ])
    dct_bbox,image = G.build_folium_png_with_gps(gis_points, 'NY.png',W=300,H=600)
    H,W = image.shape[:2]
    image = tools_draw_numpy.draw_points(image, G.gps_to_ij(gis_points,W,H,dct_bbox), color=(0, 100, 200))
    cv2.imwrite(folder_out+'NY2.png',image)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_folium_animation():
    #gis_points = numpy.array([(41.8781, -87.6298),(40.7589, -73.9851),(40.7488, -73.985428),(40.7128, -74.0060)])
    #timestamps = ["2017-04-02", "2017-06-02", "2017-07-02", "2017-08-02", "2017-09-02"]

    gis_points = numpy.array([[39.053117, -84.538639],[39.180869, -84.597492],[39.084376, -84.349243],[39.053117, -84.538639],[39.004537, -84.417423],[39.058, -84.48701],[39.053516, -84.538065],[39.03615, -84.53443],[39.053265, -84.53824],[38.99683, -84.66599],[39.036194, -84.532147]])
    map_folium = folium.Map(location=gis_points[0], tiles="stamentoner", zoom_start=5)

    # freq, period, duration = '1D', 'P1D' , 'PT1H'
    # freq, period, duration = '1H', 'PT1H', 'PT1M'
    freq,period,duration = '60s','PT1M','PT1S'

    timestamps = tools_time_convertor.datetime_to_str(pd.date_range(start=tools_time_convertor.now_str(), periods=len(gis_points), freq=freq), format='%Y-%m-%d %H:%M:%S')
    map_folium = G.add_animation(map_folium, gis_points, timestamps, period=period, duration=duration)


    map_folium.save(folder_out + 'xxxx.html')
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    tools_IO.remove_files(folder_out,'*.html,*.png,*.pdf')
    #ex_folium_simple()
    #ex_folium_complex()
    ex_folium_animation()



