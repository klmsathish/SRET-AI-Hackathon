from asyncio.windows_events import NULL
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import pandas as pd
from utils import *
from sklearn.preprocessing import normalize,MinMaxScaler
import requests
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import base64
# import pyttsx3 
# import datetime
# import speech_recognition as sr
# import threading


# def speak(audio):
#     engine = pyttsx3.init()

#     # Setting up volume level  between 0 and 1
#     engine.setProperty('volume', 1)
#     engine.setProperty('rate', 150)
#     engine.say(audio)
#     engine.runAndWait()
#     if engine._inLoop:
#         engine.endLoop()
#     # engine.iterate() must be called inside Server_Up.start()
#     # Server_Up = threading.Thread(target = __name__)
#     # Server_Up.start()
#     # engine.endLoop()
#     # quit()

# def takeCommand():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         r.pause_threshold=1
#         audio = r.listen(source)

#     try:
#         print("Recognizing...")
#         query = r.recognize_google(audio,language="en-in")
    
#     except Exception as e:
#         print(e)
#         speak("Pardon! Say that again please ")
#         return "None"
#     return query

app = Flask(__name__)
model_disease = pickle.load(open('models/model_disease.pkl', 'rb'))
model_BMI = pickle.load(open('models/model_bmi.pkl', 'rb'))
model_heart = pickle.load(open('models/model_heart.pkl','rb'))
model_stroke = pickle.load(open('models/model_stroke.pkl','rb'))
df_precaution = pd.read_csv("Preprocessed Data/Precaution.csv")
df_description = pd.read_csv("Preprocessed Data/Description.csv")
df_keys = pd.read_csv("Preprocessed Data/Keys.csv")

diseases = get_disease()


@app.route('/')
def home():
    # speak("Welcome to mr doctor!")
    return render_template('index_without_log.html', Name="Home")


@app.route('/404')
def page404():
    # speak("OOPs! You have Predicted the Unpredicted ")
    return render_template('404.html', Name="404")


@app.route('/about')
def about():
    # speak("Navigating to about page!")
    return render_template('about.html', Name="About")

@app.route('/team')
def contact():
    # speak("Navigating to about page!")
    return render_template('contact.html', Name="Team")


@app.route('/services')
def services():
    return render_template('services.html', Name="Services")


@app.route('/symptoms')
def symptoms():
    # speak("Navigating to symptom predictor!")
    return render_template('symptoms.html', diseases=diseases, Name="Symptoms")

@app.route('/addMemories')
def addMemories():
    # speak("Navigating to symptom predictor!")
    return render_template('AddMemories.html', Name="Memories")
    
@app.route("/logout")
def logout():
    return render_template("login.html", Name="logout")

@app.route("/home")
def login():
    return render_template("index.html", Name="login")

@app.route("/signup")
def signup():
    return render_template("signup.html", Name="signup")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", Name="Dashboard")

@app.route('/BMI')
def BMI():
    # speak("Navigating to BMI predictor!")
    return render_template('BMI.html', Name="BMI")
@app.route('/samples')
def sample():
    # speak("Navigating to BMI predictor!")
    return render_template('sample.html', Name="sample")

@app.route('/heart')
def heart():
    # speak("Navigating to heart disease predictor!")
    return render_template('heart.html', Name="BMI")

@app.route('/stroke')
def stroke():
    # speak("Navigating to stroke predictor!")
    return render_template('stroke.html', Name="BMI")

@app.route('/ocular')
def ocular():
    # speak("Navigating to stroke predictor!")
    return render_template('OcularDisease.html', Name="ocular")

@app.route('/ocular_predict')
def ocular_predict():
    # speak("Navigating to stroke predictor!")
    return render_template('OcularDiseasePredict.html', Name="ocular_predict")

@app.route('/covid')
def covid():
    # speak("Navigating to stroke predictor!")
    return render_template('CovidDisease.html', Name="ocular")

@app.route('/blind')
def blind():
    # speak("Navigating to stroke predictor!")
    return render_template('blind.html', Name="blind")

@app.route('/identify_objects')
def identify_objects():

    return render_template('IdentifyObjects.html', Name="identifyobjects")

@app.route('/objects_predict')
def objects_predict():
    # speak("Navigating to stroke predictor!")
    return render_template('IdentifyObjectsPredict.html', Name="objects_predict")
# @app.route('/sample')
# def sample():
#     # speak("Navigating to stroke predictor!")
#     return render_template('AddMemoriesWorking.html', Name="AddMemories")

@app.route('/upload')
def upload():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        path = "static/uploads/" + secure_filename(f.filename)
        f.save(path)
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    API_URL = "https://hf.space/embed/OFA-Sys/OFA-Image_Caption/api/predict/"

    def query(payload):
        response = requests.post(API_URL, json=payload)
        return response.json()

    output = query({
        "body": {
            "data": encoded_string,
            "example_id": NULL,
            "cleared":False,
        },
    })

    print(output)

@app.route('/BMI_predict', methods=['POST'])
def BMI_predict():
    try:
        age = request.form["age"]
        gender = request.form["gender"]
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        list = np.array([gender,age, height, weight]).reshape(1, -1)
        vals = model_BMI.predict(list)
        print("predicted:",vals)
        return render_template('BMI_predict.html',index=round(vals[0],2))
    except:
        return redirect("404")

@app.route('/symptom_predict', methods=['POST'])
def symptom_predict():
    vals = []
    pred_dict = get_pred_dict()
    for i in request.form:
        # print(request.form[i])
        vals.append(request.form[i])
    vals = [i for i in vals if i != "0"]
    if len(vals) == 0:
        return redirect("/404")
    for i in vals:
        pred_dict[i] = 1
    list = np.array([i for i in pred_dict.values()]).reshape(1, -1)
    vals = model_disease.predict(list)

    def get_data(df_keys, df_prec, df_desc, output):
        predicted_disease = df_keys[df_keys["key"] == output[0]]["disease"].to_list()[
            0]
        description = df_desc[df_desc["Disease"] ==
                              predicted_disease]["Description"].to_list()[0]
        precaution = np.array(
            df_prec[df_prec["Disease"] == predicted_disease]).tolist()[0][1:]
        return [predicted_disease, description, precaution]
    list = get_data(df_keys, df_precaution, df_description, vals)
    return render_template('symptom_predict.html', disease=list[0], description=list[1], precaution=list[2], Name="Predict")

@app.route('/heart_predict',methods=['POST'])
def heart_predict():
    try:
        Age=float(request.form['Age'])
        CigsPerDay=float(request.form['CigsPerDay'])
        Cholestrol=float(request.form['Cholestrol'])
        SysBP=float(request.form['SysBP'])
        DIaBP=float(request.form['DIaBP'])
        BMI=float(request.form['BMI'])
        HeartRate=float(request.form['HeartRate'])
        GlucoseLevel=float(request.form['GlucoseLevel'])
        Gender=float(request.form['Gender'])
        BpMedication=float(request.form['BpMedication'])
        PrevalentStroke=float(request.form['PrevalentStroke'])
        diabetes=float(request.form['diabetes'])
        list_to_be_normalised=np.array([ Age,CigsPerDay, Cholestrol, SysBP,DIaBP, BMI,HeartRate,GlucoseLevel]).reshape(1,-1)
        normalized = normalize(list_to_be_normalised)
        boolean = [Gender,BpMedication,PrevalentStroke,diabetes]
        final_features = np.append(normalized,boolean).reshape(1, -1)
        print(final_features)
        prediction = model_heart.predict(final_features)

        if prediction == 1:
            return render_template('heart_predict.html',index=1) # healthy
        else:
            return render_template('heart_predict.html',index=0) # not healthy
    except:
        return redirect("404")
# @app.route('/QA')
# def QA():
#     return render_template('QA.html', Name="BMI")
# @app.route('/QA_predict',methods=['POST'])
# def QA_predict():
#     API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
#     question = request.form['question']
#     context = request.form['context']
#     def query(payload):
#         response = requests.post(API_URL, json=payload)
#         return response.json()

#     output = query({
#         "inputs": {
#             "question": question,
#             "context": context
#         },
#     })
#     return render_template('QA_predict.html', answer=output["answer"], Name="Predict")

@app.route("/QA")
def QA():
    return render_template("QA.html")
@app.route('/QA_predict')  
def QA_predict():
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    question = request.args.get('msg')
    with open('inputs/QA_Context.txt') as f:
        lines = f.readlines()
    context = lines[0]

    def query(payload):
        response = requests.post(API_URL, json=payload)
        return response.json()
    output = query({
        "inputs": {
            "question": question,
            "context": context
        },
    })
   
    return output['answer']
  
@app.route('/stroke_predict',methods=['POST'])
def stroke_predict():
    try:
        vals = []
        pred_dict = get_stroke_dict()
        for i in list(request.form):
            # print(i)
            if i in pred_dict:
                pred_dict[i]=float(request.form[i])
            else:
                vals.append(request.form[i])
        vals = [i for i in vals if i != ""]
        for i in vals:
            pred_dict[i] = 1
        print(pred_dict.keys())
        
        features_stroke = list(pred_dict.values())
        array = np.array(features_stroke).reshape(1,-1)
        # print("Length : ", array.shape)
        
        final_features_stroke = normalize(array)
        print("Array : ",final_features_stroke)
        print(len(array))
        print(array)
        prediction_stroke = model_stroke.predict(array)
        print(prediction_stroke)
        
        if prediction_stroke == 1:
            return render_template('stroke_predict.html',index=1) # healthy
        else:
            return render_template('stroke_predict.html',index=0) # not healthy
    except:
        return redirect("404")

if __name__ == "__main__":
    app.run(debug=True, port = 8000)
    
