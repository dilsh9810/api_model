# input libraries
from flask import Flask,request,jsonify
import numpy as np
import pickle

# Load the trained model which saved

model = pickle.load(open('crop-model.model','rb'))

# Make an instance of flask api from flask-restful

app = Flask(__name__)

@app.route('/')

def index():
    return "Hello World"

@app.route('/predict',methods=['POST'])

# fetch attributes from the user inputs

def predict():
    Temperature = request.form.get('Temperature')
    Sunlight = request.form.get('Sunlight')
    PH = request.form.get('PH')
    Soil = request.form.get('Soil')
    Waterlevel = request.form.get('Waterlevel')
    Space = request.form.get('Space')

    # Get the attributes as inputs to the array
    input_query = np.array([[Temperature,Sunlight,PH,Soil,Waterlevel,Space]])

    #np.any(np.isnan(input_query))

    #np.all(np.isfinite(input_query))

    result =  model.predict(input_query)[0]

    #print the result
    return jsonify({
            'suitable crop is': str(result)
        })


    if __name__ == '__main__':
        app.run(host= '0.0.0.0')

