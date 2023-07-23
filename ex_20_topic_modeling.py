from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# ----------------------------------------------------------------------------------------------------------------------
n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20
batch_size = 128
init = "nndsvda"
# ----------------------------------------------------------------------------------------------------------------------
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    data, _ = fetch_20newsgroups(shuffle=True,random_state=1,remove=("headers", "footers", "quotes"),return_X_y=True,)
    data_samples = data[:n_samples]
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words="english")
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words="english")
    tf = tf_vectorizer.fit_transform(data_samples)

    nmf = NMF(n_components=n_components,random_state=1,init=init,beta_loss="frobenius",alpha_W=0.00005,alpha_H=0.00005,l1_ratio=1).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(nmf, tfidf_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)")

    nmf = NMF(n_components=n_components,random_state=1,init=init,beta_loss="kullback-leibler",solver="mu",max_iter=1000,alpha_W=0.00005,alpha_H=0.00005,l1_ratio=0.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(nmf,tfidf_feature_names,n_top_words,"Topics in NMF model (generalized Kullback-Leibler divergence)")

    mbnmf = MiniBatchNMF(n_components=n_components,random_state=1,batch_size=batch_size,init=init,beta_loss="frobenius",alpha_W=0.00005,alpha_H=0.00005,l1_ratio=0.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(mbnmf,tfidf_feature_names,n_top_words,"Topics in MiniBatchNMF model (Frobenius norm)")

    mbnmf = MiniBatchNMF(n_components=n_components,random_state=1,batch_size=batch_size,init=init,beta_loss="kullback-leibler",alpha_W=0.00005,alpha_H=0.00005,l1_ratio=0.5,).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(mbnmf,tfidf_feature_names,n_top_words,"Topics in MiniBatchNMF model (generalized Kullback-Leibler divergence)")

    lda = LatentDirichletAllocation(n_components=n_components,max_iter=5,learning_method="online",learning_offset=50.0,random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")