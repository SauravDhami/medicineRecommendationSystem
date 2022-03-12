from copyreg import pickle
from urllib import response
from flask import Flask,request,jsonify
from flask_cors import CORS
import Drug_Recommendation_Output


from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Welcome to CodingX</h1>"

# model = pickle.load(open('drug.pkl', 'rb'))

@app.route("/drug", methods=['POST'])
def recommend_drugs():
    medicine = request.form['med']
    res = Drug_Recommendation_Output.recommend(medicine)
    res = res[1:]
    return {"data": res}


@app.route("/check", methods=['GET'])
def check():
    return {
        "Hello": "Working"
    }
