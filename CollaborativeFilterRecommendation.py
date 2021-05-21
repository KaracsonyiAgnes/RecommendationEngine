import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

beer_colnames = ['_id', 'csvId', 'name', 'categories', 'description', 'image']
beers = pd.read_csv('datasets/beers-from-mongo.csv', usecols=beer_colnames, encoding='latin-1')
print(beers.shape)
print(beers.head())

score_colnames = ['_id', 'userId', 'beerId', 'score']
scores = pd.read_csv('datasets/scores-from-mongo.csv', usecols=score_colnames, encoding='latin-1')
print(scores.shape)
print(scores.head())

unique_users = scores.userId.unique()
unique_beers = beers._id.unique()
unique_beer_names = beers.name.unique()

n_users = unique_users.shape[0]
n_items = unique_beers.shape[0]

# change beerId to csvId
score_data_frame = pd.DataFrame(columns=['score_id', 'user_id', 'beer_id', 'score'])
for line in scores.to_numpy():
    new_line = line
    beerId = new_line[2]
    new_line[1] = np.where(unique_users == new_line[1])[0][0]
    new_line[2] = np.where(unique_beers == new_line[2])[0][0]

    if score_data_frame.index.max() >= 0:
        score_data_frame.loc[score_data_frame.index.max() + 1] = new_line
    else:
        score_data_frame.loc[0] = new_line


# create user-item matrix
data_matrix = np.zeros((n_users, n_items))
for line in score_data_frame.iterrows():
    data_matrix[line[1] - 1, line[2] - 1] = line[3]

# transpose matrix because similarity is calculated on rows
item_user_matrix = data_matrix.transpose()
similarity_matrix = 1 - pairwise_distances(item_user_matrix, metric='cosine')


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def after_work(similarity, beer_number, treshold):
    # sorting - ascending order
    sorted_similar_beers = np.argsort(similarity[beer_number])
    # top x
    top_rec = sorted_similar_beers[- treshold:]
    # descending order
    top_rec_sorted = np.flip(top_rec)

    top_beers_by_ids = []
    top_beers_by_csv_ids = []
    for item in top_rec_sorted:
        item_id = unique_beers[item]
        item_csvId = beers.loc[beers['_id'] == item_id]['csvId'].array[0]
        top_beers_by_ids.append(item_id)
        top_beers_by_csv_ids.append(item_csvId)

    return top_beers_by_ids, top_beers_by_csv_ids


result_data_frame = pd.DataFrame(columns=['id', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

for beer_name in unique_beer_names:
    sor = beers.loc[beers['name'] == beer_name]
    beer_id = sor['_id'].array[0]
    beer_csv_id = sor['csvId'].array[0]
    id_index_in_unique_beers = np.where(unique_beers == beer_id)[0][0]

    result_beer_ids = []
    recommendation_results, recommendation_results_csv = after_work(similarity_matrix, id_index_in_unique_beers, 10)
    result_beer_ids.append(recommendation_results_csv)

    if result_data_frame.index.max() >= 0:
        result_data_frame.loc[result_data_frame.index.max() + 1] = [beer_csv_id] + result_beer_ids[0]
    else:
        result_data_frame.loc[0] = [beer_csv_id] + result_beer_ids[0]

    result_data_frame.to_csv(r'C:\Users\Agi\Desktop\collaborative_filter_result_10.csv', index=False, header=True)
