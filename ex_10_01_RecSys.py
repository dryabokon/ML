#import pyodbc
import numpy
import os
import time
import cv2
import pymssql
import datetime
import multiprocessing
from collections import Counter
import pickle
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_IO
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
conn = pymssql.connect("apiqa.orgweaver.com", "d.ryabokon", "vrw308nrv2tg4", "Seksjon2")
cursor = conn.cursor()
# ----------------------------------------------------------------------------------------------------------------------
def querry_sql(sql):

    cursor.execute(sql)
    result = cursor.fetchall()
    result = numpy.array(result)

    tools_IO.write_cache(folder_out,'data.txt','',result)

    #while True:
    #    time.sleep(0.5)
    #    print('.')
    return
# ----------------------------------------------------------------------------------------------------------------------

class DB_connector(object):
    def __init__(self):
        self.cursor = self.get_cursor()
        self.temp_filename  = 'data.txt'
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_cursor(self):
        #conn = pyodbc.connect('DRIVER={SQL Server};SERVER=apiqa.orgweaver.com;DATABASE=Seksjon2;UID=d.ryabokon;PWD=vrw308nrv2tg4')
        conn = pymssql.connect("apiqa.orgweaver.com", "d.ryabokon", "vrw308nrv2tg4", "Seksjon2")
        cursor = conn.cursor()
        return cursor
    # ----------------------------------------------------------------------------------------------------------------------
    def display_console(self,result):
        for row in result:
            for value in row:
                if isinstance(value,datetime.date):
                    value = value.strftime("%H %M")
                print(value,end=' ')
            print()
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_ts_signal(self,date_stamps,values):
        signal = numpy.zeros(60*24,dtype=numpy.int)
        for date_stamp,value in zip(date_stamps,values):
            t = int(date_stamp.strftime("%H"))*60 + int(date_stamp.strftime("%M"))*1
            signal[t] = int(value)

        return signal
    # ----------------------------------------------------------------------------------------------------------------------
    def get_ts_signals(self,date_stamps, channel_ids,do_debug=False):
        dct_order = {}
        cnt =0

        dct_count = Counter(channel_ids.flatten())
        for k in dct_count.keys():
            dct_order[k]=cnt
            cnt+=1

        result = numpy.full((cnt, 60 * 24),-10)

        for date_stamp,channel_id in zip(date_stamps, channel_ids):
            id = dct_order[channel_id]
            t = int(date_stamp.strftime("%H")) * 60 + int(date_stamp.strftime("%M")) * 1
            result[id][t] = channel_id

        if 0 in dct_order:
            result = numpy.delete(result,dct_order[0],axis=0)

        if do_debug:
            print('chl  # ')
            for k,v in tools_IO.sorted_elements_by_value(dct_count,descending=True):
                print('%03d %03d'%(k,v))

        return result
    # ----------------------------------------------------------------------------------------------------------------------
    def get_daily_watch_by_memberid_MT(self,member_id,tv_date):
        date_obj = datetime.datetime.strptime(tv_date[1:-1], "%Y-%m-%d")
        weekday = date_obj.weekday()

        sql = ''' select minute, channelid from Viewing where memberid = %d and tvdate = %s''' % (member_id,tv_date)

        tools_IO.remove_file(folder_out + self.temp_filename)
        print(sql[-12:], end=' ',flush=True)
        process = multiprocessing.Process(target=querry_sql, args=(sql,))
        process.start()
        cnt=0
        while cnt<20 and not os.path.isfile(folder_out + self.temp_filename):
            time.sleep(0.5)
            cnt+=1
        process.terminate()

        result, success = tools_IO.load_if_exists(folder_out, self.temp_filename, '')
        if success and len(result) == 0:
            success = False
        if success:
            print('OK')
        else:
            print('Fail')
            return


        #display_console(result)

        signals = self.get_ts_signals(result[:,0],result[:,1],do_debug=False)
        signal  = self.get_ts_signal(result[:,0],result[:,1])
        image_signal = tools_draw_numpy.draw_signals_v2(signals,[i*60 for i in range(1,24)],tools_draw_numpy.get_colors(200,shuffle = True),w=2)

        clr = (255,100,0)
        if weekday>=5:clr = (0, 32, 255)
        cv2.putText(image_signal,'{0}'.format(date_obj.strftime("%a")),(10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1, cv2.LINE_AA)
        cv2.imwrite(folder_out+'user_%d_%s.png'%(member_id,tv_date[1:11]),image_signal)

        tools_IO.save_mat(signal,folder_out+'data_%d_%s.txt'%(member_id,tv_date[1:11]))

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_long_watchers(self,tv_date):

        print('Querry %s..' % tv_date)
        sql = ''' select top 20 memberid, count(*) as C from Viewing where tvdate = %s group by memberid order by C desc''' % (tv_date)
        self.cursor.execute(sql)
        print('OK')

        print('')
        print('  mID    # ')
        result = self.cursor.fetchall()
        self.display_console(result)

        return
    # ----------------------------------------------------------------------------------------------------------------------
    def get_period_watch_by_memberid(self,member_id,start,end):

        date_start = datetime.datetime.strptime(start[1:-1], "%Y-%m-%d")
        date_end = datetime.datetime.strptime(end[1:-1], "%Y-%m-%d")
        delta = datetime.timedelta(days=1)
        current = date_start
        range = []
        while current<=date_end:
            range.append(current)
            current+=delta

        for tv_date in range:
            the_date = '\'%4d-%02d-%02d\'' %(tv_date.year,tv_date.month,tv_date.day)
            self.get_daily_watch_by_memberid_MT(member_id, the_date)

        return
# ----------------------------------------------------------------------------------------------------------------------
tv_date_start = '\'2020-01-02\''
tv_date_end   = '\'2020-04-15\''
member_id = 1153401

'''
1067901 1286 
1236201 1208 
1131301 1179 
1041501 1103 
1057301 1066 
1088501 1031 
1010901 970 
1137001 962 
1055202 952 
1055201 952 
1252001 939 
1126602 938 
1126601 938 
1144301 936 
1129902 931 
1007401 930 
1204701 929 
1107801 916 
1153401 914 
1041702 909 

SELECT A.*,C.ContentName FROM 

(select V.tvdate AS daytime, V.memberid as memberid, V.minute, V.channelid, R.contentid, R.ProgramtypeID, R.programpart
FROM Viewing V
left JOIN Runlog R on
R.ChannelId = V.channelID
WHERE 
V.Minute BETWEEN R.StartTime AND R.EndTime
AND V.memberid = 1007401 
AND V.tvdate = '2020-03-26' 
AND R.tvdate = V.tvdate ) A
LEFT JOIN Content C ON
C.ContentId = A. contentid

ORDER BY minute
'''
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #tools_IO.remove_files(folder_out)
    C = DB_connector()
    #C.get_long_watchers(tv_date_start)
    #get_daily_watch_by_memberid(member_id,tv_date_start)

    C.get_period_watch_by_memberid(member_id,tv_date_start,tv_date_end)
    #tools_animation.folder_to_animated_gif_imageio(folder_out, folder_out+'TS.gif', mask='*.png',framerate=1.0, resize_H=512, resize_W=1440)
