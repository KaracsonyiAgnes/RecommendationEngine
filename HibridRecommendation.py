import numpy as np
import pandas as pd

wv_colnames = ['id', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eight', 'ninth', 'tenth']
wv = pd.read_csv('datasets/result_dataframe_10.csv', usecols=wv_colnames, encoding='latin-1')
# print(wv.shape)

collab_colnames = ['id', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
collab = pd.read_csv('datasets/collaborative_filter_result_10.csv', usecols=collab_colnames, encoding='latin-1')
# print(collab.shape)

result_data_frame = pd.DataFrame(columns=['id', '1', '2', '3', '4', '5'])
unique_beer_ids = collab.id.unique()

# for every beer in collab
for beer_id in unique_beer_ids:
    cf_row = collab.loc[collab['id'] == beer_id]
    cf_3 = cf_row[['2', '3', '4']]
    top_30_beers = []
    unique_30_beers = []
    for beer in cf_3.to_numpy()[0]:
        if beer < 2100:
            beer_id_str = str(beer)+'.0'
        else:
            beer_id_str = str(beer)
        wv_row = wv.loc[wv['id'] == beer_id_str]
        wv_row = wv_row[['second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eight', 'ninth', 'tenth']]
        wv_row = wv_row.to_numpy()[0]
        for item in wv_row:
            top_30_beers.append(item)
            unique_30_beers = np.unique(np.array(top_30_beers))
    cf_row = cf_row[['2', '3', '4', '5', '6', '7', '8', '9', '10']]
    cf_row = cf_row.to_numpy()[0]
    # print('intersect: ', np.intersect1d(unique_30_beers, cf_row))
    common_beers = np.intersect1d(unique_30_beers, cf_row)

    if len(common_beers) < 5:
        number_til_5 = 5 - len(common_beers)
        plus_items = unique_30_beers[:number_til_5]
    else:
        plus_items = common_beers[:5]
    result_5_beers = np.concatenate((common_beers, plus_items), axis=None)

    if result_data_frame.index.max() >= 0:
        result_data_frame.loc[result_data_frame.index.max() + 1] = np.concatenate(([beer_id], result_5_beers))
    else:
        result_data_frame.loc[0] = np.concatenate(([beer_id], result_5_beers))

    result_data_frame.to_csv(r'C:\Users\Agi\Desktop\hibrid_recommendation_5.csv', index=False, header=True)
