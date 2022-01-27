import pandas as pd
from surprise import KNNWithMeans,Dataset,Reader,SVD,accuracy
from surprise.model_selection import GridSearchCV, cross_validate,train_test_split
# ----------------------------------------------------------------------------------------------------------------------
# folder_in = './data/ex_recommend/houses/'
# filename_rating = folder_in + 'rating.csv'
# filename_product = folder_in + 'accommodation.csv'
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_recommend/movies/'
filename_rating = folder_in + 'ratings.csv'
filename_product = folder_in + 'items_raw2.csv'
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex1_cross_val(df_UIR,algrtm='KNN'):

    if algrtm=='svd':
        algo = SVD()
    else:
        sim_options = {"name": "cosine", "user_based": False}
        sim_options = {'name': 'msd', 'min_support': 3, 'user_based': False}
        algo = KNNWithMeans(sim_options=sim_options)

    data = Dataset.load_from_df(df_UIR, Reader(rating_scale=(1, 5)))

    trainset, testset = train_test_split(data, test_size=0.25)
    algo.fit(trainset)
    # prediction = algo.predict('E', 2)
    # print(prediction.est)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions,verbose=False)
    rae = accuracy.mae(predictions, verbose=False)

    print('\nrmse=%.2f\tmae=%.2f\n' % (rmse, rae))
    cross_val = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=2, verbose=False)
    print(pd.DataFrame(cross_val).head())

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex4_grid_search(df_UIR,algrtm='svd'):

    data = Dataset.load_from_df(df_UIR, Reader(rating_scale=(1, 5)))
    if algrtm=='svd':
        algo = SVD
        param_grid = {"n_epochs": [5, 10],"lr_all": [0.002, 0.005],"reg_all": [0.4, 0.6]}
        #param_grid = {"sim_options": {"name": ["msd", "cosine"], "min_support": [3, 4, 5], "user_based": [False, True]}}
    else:
        algo = KNNWithMeans
        param_grid = {'bsl_options': {'method': ['als', 'sgd'],'reg': [1, 2]},'k': [2, 3],'sim_options': {'name': ['msd', 'cosine'],'min_support': [1, 5],'user_based': [False]}}

    gs = GridSearchCV(algo, param_grid, measures=["rmse", "mae"], cv=3,joblib_verbose=0)
    gs.fit(data)

    print('\nrmse=%.2f\tmae=%.2f\n'%(gs.best_score["rmse"],gs.best_score["mae"]))

    if algrtm == 'svd':
        algo = SVD()
    else:
        algo = KNNWithMeans(sim_options=gs.best_params['rmse'])

    crs_vld = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=2, verbose=False)
    print(pd.DataFrame(crs_vld).head())

    #KNNWithMeans
    #rmse = 1.0459966783042127
    #best_params = {'bsl_options': {'method': 'als', 'reg': 1}, 'k': 3, 'sim_options': {'name': 'msd', 'min_support': 5, 'user_based': False}}

    return
# ----------------------------------------------------------------------------------------------------------------------
#df_UIR = pd.DataFrame(Dataset.load_builtin("ml-100k").raw_ratings).iloc[:, :3]
df_UIR = pd.read_csv(filename_rating, sep=',').iloc[:, :3]
#df_UIR = pd.DataFrame({"item": [1, 2, 1, 2, 1, 2, 1, 2, 1],"user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],"rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],})
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    ex1_cross_val(df_UIR)
    #ex4_grid_search(df_UIR)

