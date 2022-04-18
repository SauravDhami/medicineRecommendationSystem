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


@app.route("/drug", methods=['POST'])
def recommend_drugs():
    medicine = request.form['med']
    res = Drug_Recommendation_Output.recommend(medicine)
    res = res[1:]
    return {"data": res}

@app.route("/condition", methods=['POST'])
def recommend_Condition():
    condition = request.form['med']
    response = Drug_Recommendation_Output.recommendCondition(condition)
    response = response[:]
    return {"data": response}


@app.route("/details", methods=['POST'])
def recommend_Detail():
    medname = request.form['med']
    response = Drug_Recommendation_Output.recommend_detail(medname)
    # response = response[:]
    return {"data": response}
    


@app.route("/check", methods=['GET'])
def check():
    return {
        "Hello": "Working"
    }
