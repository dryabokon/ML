import gc
import os
import findspark
folder_spark_home = 'C:/Spark/spark-2.4.8-bin-hadoop2.7/'
findspark.init(spark_home=folder_spark_home)
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SQLContext
# ----------------------------------------------------------------------------------------------------------------------
sc = SparkContext(master="local").getOrCreate()
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def compose_path(filename_local):return os.getcwd()+filename_local
# -------------------------------------------------------------------------------   ---------------------------------------
def ex_RDD_create_seq_nums():
    rdd = sc.parallelize([1, 2, 3, 4])
    return rdd
# ----------------------------------------------------------------------------------------------------------------------
def ex_RDD_create_seq_KVP():
    rdd  = sc.parallelize([('Java', 20000), ('Python', 100000), ('Scala', 3000)])
    return rdd
# ----------------------------------------------------------------------------------------------------------------------
def ex_RDD_create_from_text():
    #rdd = sc.textFile(name="file:///C://Users//Anna//source//digits//ML//data//ex_datasets//skills.txt")
    rdd = sc.textFile(name='{prefix}{full_path}'.format(prefix='file:///', full_path=compose_path(folder_in+'skills.txt')))
    return rdd
# ----------------------------------------------------------------------------------------------------------------------
def RDD_pipelined_from_csv():
    # rdd = sc.textFile(name="file:///C://Users//Anna//source//digits//ML//data//ex_datasets//dataset_titanic.csv")
    rdd = sc.textFile(name='{prefix}{full_path}'.format(prefix='file:///', full_path=compose_path(folder_in+'dataset_titanic.csv'))).map(lambda line: line.split('\t'))

    return rdd
# ----------------------------------------------------------------------------------------------------------------------
def RDD_from_csv():       #DF->RDD
    sqlContext = SQLContext(sc)
    #RDD_titanic = sqlContext.read.csv("file:///C://Users//Anna//source//digits//ML//data//ex_datasets//dataset_titanic.csv", header=True, sep='\t').rdd
    RDD_titanic  = sqlContext.read.csv(compose_path(folder_in+'dataset_titanic.csv'), header=True,sep='\t').rdd
    return RDD_titanic
# ----------------------------------------------------------------------------------------------------------------------
def explore_RDD(rdd):

    #https://hackersandslackers.com/working-with-pyspark-rdds/

    # list all the contents
    list_of_values = rdd.collect()[:5]
    print(list_of_values)

    # list first N entries
    print(rdd.take(2))

    #
    print(rdd.count())
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #There are 4 ways of creating an RDD: from Seq or List (using Parallelize), from a text file, from another RDD, from existing DataFrames and DataSet
    # RDD_nums = ex_RDD_create_seq_nums()
    # RDD_KVP = ex_RDD_create_seq_KVP()
    # RDD_text = ex_RDD_create_from_text()


    rdd1 = RDD_pipelined_from_csv()
    explore_RDD(rdd1)

    print(''.join(['-']*20))

    rdd2 = RDD_from_csv()
    explore_RDD(rdd2)







