from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import string

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Reliable News"


@app.route('/')
def home():
    return render_template("index.html")


# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#         language = request.form("statement")
#         starwars_dictionary = {"Luke Skywalker":"1", "C-3PO":"2", "R2-D2": "3"}
#         prediction=model.predict_proba(language)
#         output='{0:.{1}f}'.format(prediction[0][1], 1)
#
#         if output== 0:
#             return render_template("index.html", pred= 'fake news')
#
#         else:
#             return render_template("index.html", pred= 'true news')
#       #  starwars_dictionary is a dictionary with character_name:character_number key-value pairs.
#        # GET URL is of the form https://swapi.co/api/people/<character_number>

@app.route("/test", methods=["GET", "POST"])
def test():
    print(f">>>>>>>>>>>>>>>>>>TEST>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    language = request.form.get("statement")
    print(language, len(language))
    if len(language) == 0:
        return render_template("index.html")
    
    else:
        # testing
        model= pickle.load(open("model.pkl",'rb'))
        vect_load = pickle.load(open("vectorizer.pickle", 'rb'))

        # preprocessing

        # vectorization = TfidfVectorizer()
        testing_news = {"statement":[language]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["statement"] = new_def_test["statement"].apply(wordopt) 
        new_x_test = new_def_test["statement"]
        new_xv_test = vect_load.transform(new_x_test)


        # test 
        predict = model.predict(new_xv_test)
        predict_label = output_lable(predict[0])
        print(f">>>>>>>>>>>>>MODEL>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model: {model}::::Predict ::::{predict}") 
        return render_template('index.html', predict_content=predict_label )


if __name__ == '__main__':
    app.run(debug=True)