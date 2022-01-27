import numpy
import pandas as pd
from surprise import KNNWithMeans,Dataset,Reader,SVD
# ----------------------------------------------------------------------------------------------------------------------
def load_product(filename_product):
    df_product = pd.read_csv(filename_product, sep=',')
    # df_product = df_product[['id','original_title']]
    # df_product = df_product.rename(columns={"original_title": "title"})
    df_product.fillna('', inplace=True)
    return df_product
# ----------------------------------------------------------------------------------------------------------------------
def generate_new_recommendation(model, user_id,df_UIR,df_product, top_n=5):
    product_titles = df_product['title'].values
    product_ids = df_product['id'].values

    current_prod_ids = df_UIR[df_UIR.iloc[:, 0] == user_id].iloc[:, 1].values

    result = []
    for product_id, product_title in zip(product_ids, product_titles):
        if product_id in current_prod_ids:
            continue
        rating = model.predict(uid=user_id, iid=product_id).est
        result.append((product_id,product_title,rating))

    result = numpy.array(result)
    result = pd.DataFrame({'product_id':result[:,0],'product_title':result[:,1],'rating':result[:,2]})
    result = result.sort_values(by='rating',ascending=False)[:top_n]

    return result
# ----------------------------------------------------------------------------------------------------------------------
def get_current_recommendations(user_id, df_UIR, df_product):
    df = df_UIR[df_UIR.iloc[:,0]==user_id]
    prod_ids = df.iloc[:,1].values
    ratings = df.iloc[:, 2].values
    result = []

    for prod_id,rating in zip(prod_ids,ratings):
        df = df_product[df_product.iloc[:,0]==prod_id]
        # if df.shape[0]<=0:
        #     continue

        title = df['title'].values[0]
        result.append((prod_id, title, rating))

    result = numpy.array(result)
    result = pd.DataFrame({'product_id': result[:, 0], 'product_title': result[:, 1], 'rating': result[:, 2]})
    result = result.sort_values(by='rating',ascending=False)

    return result
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_recommend/books/'
filename_rating = folder_in + 'ratings.csv'
filename_product = folder_in + 'books.csv'
# ----------------------------------------------------------------------------------------------------------------------
# folder_in = './data/ex_recommend/movies/'
# filename_rating = folder_in + 'ratings.csv'
# filename_product = folder_in + 'items_raw2.csv'
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
#df_UIR = pd.DataFrame({"item": [1, 2, 1, 2, 1, 2, 1, 2, 1],"user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],"rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],})
df_UIR = pd.read_csv(filename_rating, sep=',').iloc[:, :3]
# ----------------------------------------------------------------------------------------------------------------------
def ex_new_recommendations():
    algo = SVD()
    data = Dataset.load_from_df(df_UIR, Reader(rating_scale=(1, 5)))
    # cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)
    print('training OK')

    df_product = load_product(filename_product)
    user_id = 1

    df_rec0 = get_current_recommendations(user_id, df_UIR, df_product)
    print(df_rec0.to_string(index=False))
    print('--------------')

    df_rec1 = generate_new_recommendation(algo, user_id, df_UIR, df_product)
    print(df_rec1.to_string(index=False))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex_new_recommendations()