from flask import Flask, render_template, request
import pandas as pd
from tensorflow import keras
import pickle
import sklearn
import numpy as np
app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


@app.route("/prediccion", methods = ["POST"])

def predict():
    date = float(request.form.get("date"))

    votes = float(request.form.get("votes"))

    duration = float(request.form.get("duration"))

    type = float(request.form.get("type"))

    certificate = float(request.form.get('certificate'))

    nudity = float(request.form.get("nudity"))

    violence = float(request.form.get("violence"))

    profanity = float(request.form.get("profanity"))

    alcohol = float(request.form.get("alcohol"))

    frightening = float(request.form.get("frightening"))


    '''
    print('date',date)
    print('votes', votes)
    print('duration', duration)
    print('type', type)
    print('certificate', certificate)
    print('nudity', nudity)
    print('violence', violence)
    print('profanity', profanity)
    print('alchol', alcohol)
    print('fritering', frightening)
    '''
    data = [[date, votes, duration, type, certificate, nudity, violence, profanity, alcohol, frightening]]
    columnas = ['Date','Votes','Duration','Type','Certificate','Nudity','Violence','Profanity','Alcohol','Frightening']

    df = pd.DataFrame()
    #df = df.assign(Date = None, Votes = None, Duration = None, Type = None, Certidicate = None, Nudity = None, Violence = None,Profanity= None,
     #              Alcohol = None,Frightening=None)

   # data = {'Date': date, 'Votes':votes, 'Duration': duration, 'Type':type, 'Certificate': certificate,
    #        'Nudity':nudity, 'Violence':violence, 'Profanity':profanity, 'Alcohol':alcohol,'Frightening': frightening}

    df = pd.DataFrame(data, columns= columnas)
    print(df.info())
    print(df)
    #Cargar random forest
    randomForest = pickle.load(open('random_forest.sav', 'rb'))
    #Cargar regresion lineal
    regresionLineal = pickle.load(open('linear_regression.sav', 'rb'))
    #Cargar SVR
    svr = pickle.load(open('svr.sav', 'rb'))
    #Cargar knn
    knn = pickle.load(open('knn.sav', 'rb'))
    #Cargar Escalado
    scaler = pickle.load(open("scaler.sav", 'rb'))
    df = df.values
    test_scaled_set = scaler.transform(df)
    test_scaled_set = df
    #print(test_scaled_set)




    resultado_float = np.round(randomForest.predict(test_scaled_set),2)
    #print(resultado_float)
    resultado_random = resultado_float[0].astype('str')
    resultado_regresion = np.round(regresionLineal.predict(test_scaled_set),2)
    resultado_regresion = resultado_regresion[0].astype('str')
    resultado_svr = np.round(svr.predict(test_scaled_set),2)
    resultado_svr = resultado_svr[0].astype('str')
    resultado_knn = np.round(knn.predict(test_scaled_set), 2)
    resultado_knn = resultado_knn[0].astype('str')


    #,  resultado_str =  resultado_str resultado_svr = resultado_svr,    resultado_knn =  resultado_knn
    return render_template("prediccion.html",  resultado_random =  resultado_random,  resultado_regresion = resultado_regresion,
                           resultado_svr=resultado_svr, resultado_knn=resultado_knn )

if __name__ == '__main__':
    app.run()
