
import pandas as pd
import pickle
from flask import Flask, render_template, request



app = Flask(__name__)

data = pd.read_csv('cleaned_data_house_price.csv')
pipe = pickle.load(open('Ridgemodel.pkl', 'rb'))


@app.route('/')


def index():
    locations = sorted(data['location'].unique())

    return render_template('index.html',locations = locations)



@app.route('/predict',methods = ['POST'])

def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # bhk = float(bhk)
    # bath = float(bath)
    print(location, bhk, bath, sqft)

    input = pd.DataFrame([[location,sqft,bath,bhk]], columns = ['location','total_sqft','bath','bhk'])

    # print(input)

    prediction = pipe.predict(input)[0] * 100000

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)