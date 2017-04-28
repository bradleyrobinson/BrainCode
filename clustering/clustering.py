"""

"""
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure, save, output_file
import numpy as np
import os
import pandas as pd
from scipy.spatial import distance


def get_colors(label):
    if label == 'Centroid':
        return 'yellow'
    elif label == 'Cluster 1':
        return 'blue'
    else:
        return 'red'


def make_graph(points, title):
    x = points['yeas'].tolist()
    y = points['nays'].tolist()
    names = points['names'].tolist()
    print(len(x), len(y), len(names))
    source = ColumnDataSource(data=dict(yeas=x, nays=y, cluster=points['labels'], names=names))
    colors = [get_colors(label) for label in points['labels'].values]
    hover = HoverTool(
        tooltips=[
            ("Point Type", "@cluster"),
            ("Name", "@names"),
            ("Yeas", "@yeas"),
            ("Nays", "@nays")
        ]
    )
    p = figure(title="", plot_width=600, plot_height=400)
    p.add_tools(hover)
    p.scatter(x, y, fill_color=colors, line_color="black", source=source, alpha=.7, size=12)
    output_file(os.path.join('clustering_demo', title + '.html'))
    save(p)


def step_0(points):
    labels = np.array(['None' for _ in range(points.shape[0])])
    points['labels'] = labels
    make_graph(points, "Before Clustering")


def step_1(points, k):
    """
    Picks random points as the centroids, creates a graph for visualization, returns the data

    Parameters:
        points:
        k:

    Returns:
    """
    indices = np.random.choice(np.arange(points.shape[0]), size=k, replace=False)
    points.ix[indices, 'labels'] = "Centroid"
    centroid_locations = points[['yeas', 'nays']].iloc[indices]
    make_graph(points, "Random Centroids")
    return centroid_locations


def find_row_cluster(row):
    distances = [distance.euclidean([row[1], row[2]], centroids[0,:]),
                 distance.euclidean([row[1], row[2]], centroids[1,:])]
    if distances[0] > distances[1]:
        return "Cluster 1"
    else:
        return "Cluster 2"


def find_cluster(points, i):
    points.ix[:, 'labels'] = points.apply(find_row_cluster, axis=1)
    make_graph(points, "Clusters found iteration {}".format(i))


def average_point(cluster_points):
    average_x = cluster_points['yeas'].mean()
    average_y = cluster_points['nays'].mean()
    return average_x, average_y


def compute_new_centroids(points):
    cluster_1 = average_point(points[points['labels'] == "Cluster 1"])
    cluster_2 = average_point(points[points['labels'] == "Cluster 2"])
    return np.array([cluster_1, cluster_2])


def get_data():
    pre_processed_data = pd.read_csv(os.path.join('data', 'H2017_voting.csv'), index_col=1)
    pre_processed_data = pre_processed_data.drop(pre_processed_data.columns[0:1], axis=1)
    representatives = pre_processed_data.index.tolist()
    house = pre_processed_data.transpose()
    voting_record = []
    for r in representatives:
        yes_votes = house[house[r] == 1].shape[0]
        no_votes = house[house[r] == 0].shape[0]
        voting_record.append([r, yes_votes, no_votes])
    columns = ['names', 'yeas', 'nays']
    votes = pd.DataFrame(voting_record, columns=columns)
    return votes


if __name__ == '__main__':
    votes = get_data()
    step_0(votes)
    # TODO: make this less confusing
    centroids = np.array(step_1(votes, 2).ix[:, [0, 1]].values) # Random values
    for i in range(10):
        find_cluster(votes, i)
        centroids = compute_new_centroids(votes)
        centroid_named = pd.DataFrame({"names": ["Cluster 1", "Cluster 2"], "labels": ["Centroid", "Centroid"],
                                       'yeas': centroids[:,0], 'nays': centroids[:,1]})
        make_graph(votes.append(centroid_named).reset_index(drop=True), "Iteration {} with centroids".format(i))




