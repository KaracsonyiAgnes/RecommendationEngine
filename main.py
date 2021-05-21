from flask import Flask, request, jsonify, abort
import pandas as pd

app = Flask(__name__)

@app.route("/flask")
def index():
    return jsonify("Recommendation API")

@app.errorhandler(401)
def api_key_not_found(e):
    return jsonify(error=str(e)), 401

@app.errorhandler(422)
def json_not_found(e):
    return jsonify(error=str(e)), 422


@app.route("/flask/recommendation", methods=['GET', 'POST'])
def extract_keywords():
    req_data = request.get_json()

    if not ('beerId' in req_data):
        abort(422, description="beer id attribute is not found")

    if not req_data['beerId']:
        abort(422, description="beer id attribute has no value")

    if ('beerId' in req_data) and req_data['beerId']:
        beer_id = req_data['beerId']
    else:
        beer_id = 1

    print('beerId: ', beer_id, ' type: ', type(beer_id))
    beer_colnames = ['id', '1', '2', '3', '4', '5']
    beers = pd.read_csv('datasets/hibrid_recommendation_5.csv', usecols=beer_colnames, encoding='latin-1')

    selected_beers = beers.loc[beers['id'] == beer_id]
    print(selected_beers)
    js = selected_beers.to_json(orient='records')
    print(js)
    return js, 200


if __name__ == '__main__':
    app.run(host='localhost', port=80)
