#Importing Required Libraries
from flask import Flask,render_template,request
import numpy as np
import pickle

#Initializing
app = Flask(__name__)

#Home page for the Application
@app.route('/')
@app.route('/home')
def home():
    return render_template("H.html")

#Url for Regression analysis
@app.route('/R')
def R():
    return render_template("R.html")

#Url for Classification
@app.route('/C')
def C():
    return render_template("C.html")

#To handle the FORM data and use those fields for MonthlyIncome prediction
@app.route("/submit",methods=["POST","GET"])
def submit():
    if request.method=='POST':
        record=[1]
        Fields = ['JL','TWH','DPT','YWCM','JR','AGE','DFH']
        for i in Fields:
            record.append(float(request.form[i]))
        loaded_model = pickle.load(open("Linear_model.sav", 'rb'))
        prediction = loaded_model.predict(record)
        return render_template("M.html",pred=str(round(prediction[0],2)))
    
#To handle the FORM data and use those fields for Attrition prediction
@app.route("/submit1",methods=["POST","GET"])
def submit1():
    if request.method=='POST':
        record1=[]
        Fields1 = ['AGE','JL','JR','WLB','YAC','YSLP']
        for i in Fields1:
            record1.append(float(request.form[i]))
        loaded_model1 = pickle.load(open("DT_model.sav", 'rb'))
        prediction1 = loaded_model1.predict(np.asarray(record1).reshape(1,-1))
        return render_template("CH.html",pred=str(np.where(prediction1[0]==1,"Yes","No")))

if __name__=='__main__':
    app.run(port=3000)