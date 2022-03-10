from copyreg import pickle
from urllib import response
from flask import Flask,request,jsonify
from flask_cors import CORS
import Drug_Recommendation_Output
import pickle
import json

app = Flask(__name__)
# CORS(app) 
model = pickle.load(open('drug.pkl', 'rb'))

        
@app.route('/drug', methods=['POST'])
def recommend_drugs():
    # res = Drug_Recommendation_Output.recommend(request.args.get(''))
    medicine = request.form['med']
    # medicine = request.form.values
    # print(medicine)
 
    res = Drug_Recommendation_Output.recommend(medicine)
    # res = model.recommend(medicine)
  
    # print(res)
    
    # return json.dumps(res)
    return {"data": res}
    


if __name__=='__main__':
    app.run(port=8888, debug=True)

