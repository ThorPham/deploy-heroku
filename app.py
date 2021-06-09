import pandas as pd
import numpy as np 
import sklearn
import joblib 
from flask import Flask , render_template, request 

app = Flask(__name__,template_folder='templates')
# declare model prediction
model=open("linear_regression_model.pkl","rb")
lr_model=joblib.load(model)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # print(request.form)
    if request.method == 'POST':
        var_1 = request.form['var_1']
        var_2 = request.form['var_2']
        var_3 = request.form['var_3']
        var_4 = request.form['var_4']
        var_5 = request.form['var_5']
        input_preprocesor = np.array([var_1,var_2,var_3,var_4,var_5],dtype=np.float32)
        tmp = input_preprocesor.reshape(1,-1)
        model_pred = lr_model.predict(tmp)
    return render_template('predict.html',prediction=model_pred)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000', use_reloader=False)