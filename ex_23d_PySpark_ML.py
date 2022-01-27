import cv2
import time
import shutil
import os
import numpy
import pandas as pd
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import argparse
from pyspark.sql import SparkSession
# ----------------------------------------------------------------------------------------------------------------------
def init_spark():
    try: spark
    except NameError: spark = SparkSession.builder.getOrCreate()
    try:sc
    except NameError: sc = spark.sparkContext
    sc.setLogLevel('WARN')

    return spark,sc
# ----------------------------------------------------------------------------------------------------------------------
spark, sc = init_spark()
# ----------------------------------------------------------------------------------------------------------------------
def remove_folders(path):

    if (path==None):return
    if not os.path.exists(path):return

    filelist = [f for f in os.listdir(path)]
    for f in filelist:
        if os.path.isdir(path + f):
            shutil.rmtree(path + f)

    shutil.rmtree(path)
    return
# ----------------------------------------------------------------------------------------------------------------------
def pipeline_train(filename_rating,folder_out):

    # Ratings should be a structure of 3 columns: userID, productID, rating
    dfRates = spark.createDataFrame(pd.read_csv(filename_rating, sep=',').iloc[:,:3])
    sc.setCheckpointDir(folder_out + 'checkpoint/')
    model = ALS.train(ratings=dfRates.rdd, rank=5)

    return model
# ----------------------------------------------------------------------------------------------------------------------
def pipeline_predict(model,filename_rating,filename_product,n_top=3):

    # Ratings should be a structure of 3 columns: userID, productID, rating
    # Product should be a structure of N columns: poductID, ...

    filename_out = folder_out + 'pred.txt'
    dfRates = spark.createDataFrame(pd.read_csv(filename_rating, sep=',').iloc[:,:3])
    dfPrdct = spark.createDataFrame(pd.read_csv(filename_product, sep=','))
    user_ids = numpy.sort(numpy.array([x[0] for x in dfRates.select(dfRates.columns[0]).distinct().collect()]))

    if os.path.isfile(filename_out):os.remove(filename_out)

    for user_id in user_ids:
        df_rated_products     = dfRates.rdd.filter(lambda x: x[0] == int(user_id)).map(lambda r: r[1]).collect()
        df_not_rated_products = dfPrdct.rdd.filter(lambda x: x[0] not in df_rated_products)
        rdd_input = df_not_rated_products.map(lambda x: (user_id, x[0]))

        predictions = model.predictAll(rdd_input)
        predictions = predictions.map(lambda p: (str(p[0]), str(p[1]), float(p[2])))
        predictions = predictions.takeOrdered(n_top, key=lambda x: -x[2])
        append_predictions(predictions,filename_out)
        print(user_id)

    return
# ----------------------------------------------------------------------------------------------------------------------
def append_predictions(predictions, filename_out):
    A = numpy.array(predictions)
    df_pred = pd.DataFrame({'userID':A[:,0],'prodID':A[:,1],'rating':A[:,2]})
    if not os.path.isfile(filename_out):
        df_pred.to_csv(filename_out,index=False)
    else:
        df_pred.to_csv(filename_out,index=False, mode='a', header=False)

    return
# ----------------------------------------------------------------------------------------------------------------------
def enrich_ratings(filename_rating, filename_product,folder_out):

    dfRates = pd.read_csv(filename_rating, sep=',').iloc[:,:3]
    dfPrdct = pd.read_csv(filename_product, sep=',')
    dfPrdct = dfPrdct.rename(columns={dfPrdct.columns[0]: dfRates.columns[1]})

    how = 'right'
    key = dfPrdct.columns[0]
    df_result = pd.merge(dfRates, dfPrdct, how=how, on=[key])
    df_result.to_csv(folder_out+'pred_ext.csv', index=False)

    return
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_recommend/houses/'
filename_rating = folder_in + 'rating.csv'
filename_product = folder_in + 'accommodation.csv'
# ----------------------------------------------------------------------------------------------------------------------
# folder_in = './data/ex_recommend/movies/'
# filename_rating = folder_in + 'ratings.csv'
# filename_product = folder_in + 'items_raw2.csv'
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def E2E_train_test(filename_rating,filename_product,folder_out):

    if not os.path.exists(folder_out):
         os.mkdir(folder_out)

    folder_model = folder_out + 'ALS_model'

    model = pipeline_train(filename_rating,folder_out)
    remove_folders(folder_model)
    model.save(sc, folder_model)

    model = MatrixFactorizationModel.load(sc, folder_model)
    pipeline_predict(model,filename_rating, filename_product)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', default='predict')
    parser.add_argument('--filename_rating', default=filename_rating)
    parser.add_argument('--filename_product', default=filename_product)
    parser.add_argument('--folder_out', default=folder_out)
    args = parser.parse_args()

    if args.command=='predict':
        time_start = time.time()
        E2E_train_test(args.filename_rating, args.filename_product,args.folder_out)
        print('Execution time: %.2fs'%(time.time()-time_start))

    if args.command=='enrich':
        enrich_ratings(args.filename_rating, args.filename_product,args.folder_out)
