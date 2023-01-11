import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def optimal_number_clusters(df):
    """
    Calculates optimal number of clusted based on Elbow Method
    
    parameters df
    """
    
    Sum_of_squared_distances = []
    K = range(2, 20) # define the range of clusters we would like to cluster the data into

    for k in K:
        km = KMeans(n_clusters = k)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)

    plt.figure(figsize=(20,10))

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.xticks(K)
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show();
    
    
    
def visualize_clusters(y_kmeans, data_scaled, n_clusters):
    """
    Visualize the users with predicted clusters.

    Run PCA on the transposed data and reduce the dimnensions in pca_num_components dimensions

    """

    reduced_data = PCA(n_components = 2).fit_transform(data_scaled)
    results = pd.DataFrame(reduced_data, columns = ['pca1','pca2'])
    results = pd.concat([results, pd.DataFrame(y_kmeans)], axis = 1).rename(columns={0 : "cluster"})

    cmap = sns.color_palette("Set1", n_colors = n_clusters, desat = .5)

    sns.scatterplot(x = 'pca1', y = 'pca2', hue = 'cluster', data = results, palette = cmap, legend = True)
    plt.title('K-means Clustering with 2 dimensions')
    plt.show();