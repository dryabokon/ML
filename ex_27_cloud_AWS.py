import numpy
import pandas as pd
import uuid
# ----------------------------------------------------------------------------------------------------------------------
import tools_AWS
import tools_time_profiler
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
T = tools_time_profiler.Time_Profiler()
# ----------------------------------------------------------------------------------------------------------------------
filename_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
S3 = tools_AWS.processor_S3(filename_config='./config.private', folder_out=filename_out)
schema_name_in = 'activegps'
schema_name_out = 'activegps_pipelines'
# ----------------------------------------------------------------------------------------------------------------------
def ex_explore_S3():
    S3.get_all_buckets_stats(verbose=True)
    S3.get_bucket_stats(verbose=True)
    #S3.create_bucket('name_of_bucket')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_upload_file_s3():
    S3.upload_file('./data/ex_datasets/dataset_titanic.csv',aws_file_key='dataset_titanic.csv')
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_delete_file_s3():
    S3.get_bucket_stats(verbose=True)
    S3.delete_file('cars/temp_table_a0acc32d68254d7ab642c9a9b119c9fa/20220322_092651_00068_tu2xj_1ddd5e9c-6106-43ae-b757-92dfb6dff1cd')
    S3.get_bucket_stats(verbose=True)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_delete_files_s3():
    S3.get_bucket_stats(verbose=True)
    S3.delete_files_by_prefix(aws_file_prefix='1')
    S3.delete_all_files()
    S3.get_bucket_stats(verbose=True)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_explore_athena():
    table_name = 'v_carmodel'

    S3.get_databases(verbose=True)
    S3.get_tables(schema_name_in,verbose=True)
    S3.get_table(schema_name_in,table_name,verbose=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_create_database():
    database_name = 'activegps_pipelines'
    S3.execute_query('CREATE SCHEMA IF NOT EXISTS %s' % (database_name),schema_name=None)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_create_mat_view():
    table_name_in = 'v_carmodel'
    table_name_out = 'v_delme3'

    S3.create_view(schema_name_out, table_name_out, "select * from %s.%s limit 6" % (schema_name_in, table_name_in))
    S3.get_tables(schema_name_out,verbose=True)
    #S3.get_bucket_stats(verbose=True,idx_sort=2)
    S3.get_table(schema_name_out,table_name_out, verbose=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_create_table():

    rows, cols = 5, 3
    idx_dates = pd.date_range("20210101", periods=rows)
    df = pd.DataFrame({'t': numpy.arange(0, rows),
                       'key': [(chr(65 + int(a))) for a in numpy.random.rand(rows) * 25],
                       'value': [int(a) for a in numpy.random.rand(rows) * 10]}, index=idx_dates)

    #df = pd.DataFrame({'A':['a','b'],'B':['cc','dd']})
    #print(tools_DF.prettify(df))

    table_name_out = 't_temp'

    S3.drop_table(schema_name_out, table_name_out)
    #S3.get_tables(schema_name_out, verbose=True)
    S3.create_table_from_df_v2(schema_name_out,table_name_out,df)
    #S3.get_tables(schema_name_out, verbose=True)
    S3.get_table(schema_name_out, table_name_out, verbose=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_drop_table():
    S3.get_tables(schema_name_out, verbose=True)
    S3.drop_table(schema_name_in, 'v_tmp_users')
    S3.get_tables(schema_name_out, verbose=True)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_execute_query_to_df(method = 1):
    T.tic('ex_execute_query_to_df_method_%d'%method)
    table_name_in = 'v_carmodel'
    SQL = "SELECT * from %s.%s limit 10" % (schema_name_in, table_name_in)

    if method ==1:
        df = S3.get_table(schema_name_in, table_name_in,verbose=True)
    elif method==2:
        df = S3.execute_query_wr(SQL, schema_name=schema_name_in,verbose=True)
    else:
        aws_file_key = S3.execute_query(SQL,schema_name=schema_name_in,aws_file_key_out=uuid.uuid4().hex)
        df = S3.downdload_df(aws_file_key)

    df.to_csv(S3.folder_out + table_name_in + '.csv', index=False)
    T.print_duration('ex_execute_query_to_df_method_%d'%method)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_execute_query_to_S3(method = 1):
    T.tic('ex_execute_query_to_S3_method_%d'%method)
    table_name_in = 'v_carmodel'
    aws_file_key_out = 'cars'

    SQL = "select * from %s.%s limit 10" % (schema_name_in, table_name_in)

    S3.get_bucket_stats(verbose=True)
    aws_file_key = S3.execute_query(SQL,schema_name=schema_name_in,aws_file_key_out=aws_file_key_out)
    S3.get_bucket_stats(verbose=True)
    if method == 1:
        S3.downdload_file(S3.folder_out+'xxx.csv',aws_file_key=aws_file_key)
    else:
        df = S3.downdload_df(aws_file_key)
        df.to_csv(S3.folder_out+'yyy.csv',index=False)

    T.print_duration('ex_execute_query_to_S3_method_%d'%method)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_execute_query_to_S3(method=1)
    #ex_create_mat_view()
    #ex_create_pivot()
    ex_create_table()

    # S3.delete_all_files()
    #S3.get_bucket_stats(verbose=True)