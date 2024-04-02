from flask import Flask , request,jsonify
import numpy as np
from flask_cors import CORS
import pandas
import pickle
import sklearn
import json
# loading model 
dtr=pickle.load(open('dtr.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))
#global varible
prediction = ''

#creating flask app
app = Flask (__name__)
# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api',methods=['GET','POST'])
def nameRoute():
    global prediction
    if(request.method=='POST'):
        request_data=request.data
        request_data=json.loads(request_data.decode('utf-8'))
        Year=request_data['Year']
        average_rain_fall_mm_per_year = request_data['average_rain_fall_mm_per_year']
        pesticides_tonnes = request_data['pesticides_tonnes']
        avg_temp = request_data['avg_temp']
        Area = request_data['Area']
        Item  = request_data['Item']
        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1).tolist()
        return ""
    else:
        return jsonify({'prediction':prediction})
    