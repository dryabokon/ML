import os
import findspark
folder_spark_home = 'C:/Spark/spark-2.4.8-bin-hadoop2.7/'
findspark.init(spark_home=folder_spark_home)
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SQLContext
sc = SparkContext(master="local").getOrCreate()
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def compose_path(filename_local):return os.getcwd()+filename_local
# ----------------------------------------------------------------------------------------------------------------------
#https://www.nbshare.io/notebook/97969492/Data-Analysis-With-Pyspark-Dataframe/
# ----------------------------------------------------------------------------------------------------------------------
def ex_DF_create_from_csv():
    sqlContext = SQLContext(sc)
    #DF_titanic = sqlContext.read.csv("file:///C://Users//Anna//source//digits//ML//data//ex_datasets//dataset_titanic.csv", header=True, sep='\t')
    DF_titanic  = sqlContext.read.csv(compose_path(folder_in+'dataset_titanic.csv'), header=True,sep='\t')

    # list_of_values = RDD_titanic.map(lambda line: line.split('\t')).collect()       # this is list
    # RDD_titanic = SQLContext(sc).createDataFrame(list_of_values[1:])                # this is DataFrame

    return DF_titanic
# ----------------------------------------------------------------------------------------------------------------------
def ex_DF_create_from_csv_unstable():
    RDD_text = sc.textFile(name='{prefix}{full_path}'.format(prefix='file:///', full_path=compose_path(folder_in + 'dataset_titanic.csv')))
    RDD_text  = RDD_text.map(lambda line: line.split('\t'))  # this is RDD
    DF_text = RDD_text.toDF()
    return DF_text
# ----------------------------------------------------------------------------------------------------------------------
def explore_DF(df):
    df.printSchema()
    df.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    df = ex_DF_create_from_csv()
    explore_DF(df)