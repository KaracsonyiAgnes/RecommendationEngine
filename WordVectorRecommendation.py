import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import gensim.downloader


def load_data():
    beer_colnames = ['id', 'name', 'cat_id', 'descript']
    beers = pd.read_csv('datasets/beers.csv', usecols=beer_colnames, encoding='latin-1')
    print(beers.shape)
    print(beers.head())

    beers_with_descriptions = beers.dropna(axis=0, subset=['descript'])
    print(beers_with_descriptions.shape)
    print(beers_with_descriptions.head())

    data_with_filtered_columns = beers_with_descriptions[['id', 'name', 'descript']]
    return data_with_filtered_columns


def get_every_description(data_with_filtered_columns):
    every_description = [row.split() for row in data_with_filtered_columns['descript']]
    return every_description


def train_model():
    model = gensim.downloader.load('glove-twitter-200')
    return model


def get_unique_beer_names(data_with_filtered_columns):
    unique_beer_names = list(data_with_filtered_columns.name.unique())
    return unique_beer_names


def generate_mean_of_word_vectors_for_every_description(model, every_description):
    mean_vectors = {}
    num_of_description = 0
    for one_description in every_description:
        num_of_description += 1
        vectors = []
        for word in one_description:
            try:
                vectors.append(model[word] / np.linalg.norm(model[word]))
            except:
                pass
        mean_vector = np.mean(vectors, axis=0)
        mean_vectors[num_of_description] = mean_vector
    return mean_vectors


def cosine_distance(model, description, data_with_filtered_columns, mean_vectors, num):
    cosine_dict = {}
    word_list = []
    vectors = []
    for word_of_description in description.split():
        try:
            vectors.append(model[word_of_description] / np.linalg.norm(model[word_of_description]))
        except:
            pass
    mean_of_descriptions_word_vectors = np.mean(vectors, axis=0)
    num_of_description = 0
    for row in data_with_filtered_columns.to_numpy():
        beer_index = row[0]
        one_description = [row[2].split()]
        num_of_description += 1
        if one_description != description:
            b_vectors = mean_vectors.get(num_of_description)
            if type(mean_of_descriptions_word_vectors) == np.float64 or type(b_vectors) == np.float64:
                cos_sim = 0
            else:
                cos_sim = dot(mean_of_descriptions_word_vectors, b_vectors) / (norm(mean_of_descriptions_word_vectors) * norm(b_vectors))
            cosine_dict[beer_index] = cos_sim
    dist_sort = sorted(cosine_dict.items(), key=lambda dist: dist[1], reverse=True)  # in Descending order
    for one_description in dist_sort:
        if one_description != description:
            word_list.append((one_description[0], one_description[1]))
    return word_list[0:num]


# MAIN

result_dataframe = pd.DataFrame({'id': [],
                                 'first': [],
                                 'second': [],
                                 'third': [],
                                 'fourth': [],
                                 'fifth': [],
                                 'sixth': [],
                                 'seventh': [],
                                 'eight': [],
                                 'ninth': [],
                                 'tenth': []})

data_with_filtered_columns = load_data()
every_description = get_every_description(data_with_filtered_columns)
model = train_model()

# model.to_csv(r'C:\Users\Agi\Desktop\model.csv', index=False, header=True)
# model_colnames = ['id', 'name', 'cat_id', 'descript']
# model_new = pd.read_csv('datasets/model.csv', usecols=model_colnames, encoding='latin-1')

mean_vectors = generate_mean_of_word_vectors_for_every_description(model, every_description)
unique_beer_names = get_unique_beer_names(data_with_filtered_columns)

for beer_name in unique_beer_names:
    sor = data_with_filtered_columns.loc[data_with_filtered_columns['name'] == beer_name]
    desc = sor['descript'].array[0]
    beer_id = sor['id'].array[0]
    result = cosine_distance(model, desc, data_with_filtered_columns, mean_vectors, 10)

    result_beer_names = []
    for result_beer in result:
        result_id = result_beer[0]
        beer_row = data_with_filtered_columns.loc[data_with_filtered_columns['id'] == result_id]
        if len(beer_row['id'].array) > 0:
            beer_row = beer_row['id'].array[0]
        else:
            beer_row = ''
        result_beer_names.append(beer_row)

    if result_dataframe.index.max() >= 0:
        result_dataframe.loc[result_dataframe.index.max() + 1] = [beer_id] + result_beer_names
    else:
        result_dataframe.loc[0] = [beer_id] + result_beer_names

    result_dataframe.to_csv(r'C:\Users\Agi\Desktop\result_dataframe_10.csv', index=False, header=True)
