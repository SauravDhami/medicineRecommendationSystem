from copyreg import pickle
from urllib import response
from flask import Flask,request,jsonify
from flask_cors import CORS
import Drug_Recommendation_Output
import traceback



from flask import Flask
app = Flask(__name__)


@app.errorhandler(500)
def internal_error(exception):
    print(traceback.format_exc())

@app.route("/")
def index():
    return "<h1>Welcome to MRS System.</h1>"


@app.route("/drug", methods=['POST'])
def recommend_drugs():
    medicine = request.form['med']
    print(medicine)
    res = Drug_Recommendation_Output.recommend(medicine)
    res = res[:]
    return {"data": res}

@app.route("/condition", methods=['POST'])
def recommend_Condition():
    condition = request.form['con']
    print(condition)
    responseCondition = Drug_Recommendation_Output.recommendCondition(condition)
    responseCondition = responseCondition[:]
    print(responseCondition)
    return {"data": responseCondition}


@app.route("/details", methods=['POST'])
def recommend_Detail():
    medname = request.form['det']
    print(medname)
    responseDetail = Drug_Recommendation_Output.recommend_detail(medname)
    # response = response[:]
    return {"data": responseDetail}
    


@app.route("/check", methods=['GET'])
def check():
    return {
        "Hello": "Working"
    }
