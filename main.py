import nltk 
nltk.download('punkt')
from copyreg import pickle
from urllib import response
from flask import Flask,request,jsonify
from flask_cors import CORS
# from Drug_Recommendation_Output import recommend
import Drug_Recommendation_Output
import pickle
import json



# app = Flask(name)
# # CORS(app) 
# model = pickle.load(open('drug.pkl', 'rb'))


# @app.route('/drug', methods=['POST'])
# def recommend_drugs():
#     # res = Drug_Recommendation_Output.recommend(request.args.get(''))
#     medicine = request.form['med']
#     # medicine = request.form.values
#     # print(medicine)
 
#     res = Drug_Recommendation_Output.recommend(medicine)
#     # res = model.recommend(medicine)

#     # print(res)

#     # return json.dumps(res)
#     return {"data": res}

# @app.route('/check', methods=['GET'])
# def check():
#     return {
#         "Hello": "Working"
#     }


# if name=='main':
#     app.run(port=8888, debug=True)

from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Welcome to CodingX</h1>"

# model = pickle.load(open('drug.pkl', 'rb'))

@app.route('/drug', methods=['POST'])
def recommend_drugs():
    medicine = request.form['med']
    res = Drug_Recommendation_Output.recommend(medicine)
    return {"data": res}


@app.route('/check', methods=['GET'])
def check():
    return {
        "Hello": "Working"
    }

# if __name__ == "__main__":
#   app.run()