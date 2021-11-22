import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

option = st.sidebar.selectbox(
    'Select one:',
     ['lntroduction','Analysis', 'Conclusion','The End'])

if option=='lntroduction':
    
    st.header("Heart Attack Analysis")
    st.write("Heart is very important for human.")
    st.write("Taking care of it may save our life.")   

    
    img = Image.open("heart.jpg")
    st.image(img, use_column_width = True)


    from PIL import Image
    #imgg = Image.open("heart_attack.jpg")
    #st.image(imgg)

    st.write("- The Heart Analysis Dataset:")
    
    path = '/content/heart.csv'
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df.head(10)
    st.dataframe(df)


elif option=='Analysis':

    st.write("- The analysis is done using different kind of supervised machine learning.")
    path = '/content/heart.csv'
    df = pd.read_csv(path)
    X_heart = df.drop("output", axis=1)
    y_heart = df["output"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.3, random_state=42)
    models = pd.DataFrame(columns=["Model","Accuracy Score"])
    
    # LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    predictions = log_reg.predict(X_test)
    score1 = accuracy_score(y_test, predictions)
    st.write("Logistic Regression is equal to", score1)

    # GAUSSIAN NAIVE BAYES
    from sklearn.naive_bayes import GaussianNB
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)
    predictions = GNB.predict(X_test)
    score2 = accuracy_score(y_test, predictions)
    st.write("GaussianNB is equal to", score2)

    # KNN CLASSIFIER 
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    score3 = accuracy_score(y_test, predictions)
    st.write("KNeighborsClassifier is equal to", score3)

    # SVM
    from sklearn.svm import SVC
    svm = SVC(random_state=0)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    score4 = accuracy_score(y_test, predictions)
    st.write("SVM is equal to", score4) 

    # RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    randomforest = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42)
    randomforest.fit(X_train, y_train)
    predictions = randomforest.predict(X_test)
    score5 = accuracy_score(y_test, predictions)
    st.write("Random Forest Classifier is equal to", score5) 

    st.write("- The score can be simplify as shown in the table below:")
    st.write(pd.DataFrame({
      '': ['Logistic Regression', 'GaussianNB', 'KNeighborsClassifier', 'SVM', 'Random Forest Classifier'],
      'Score': [score1, score2, score3, score4, score5]
    }))  

    st.write("- From all the scores, the accuracy of all the outputs from the model are between 66 percent to 83 percent.")

elif option=='Conclusion':
   st.write("Always practicing good healthy lifestyle.")
   st.write("Healthy lifestyle will promote a healthy heart.")

   img = Image.open("h.jpg")
   st.image(img, use_column_width = True)

   st.title("HEALTHY LIFE, HEALTHY HEART")
    
else:
    'Starting a long computation...'
    
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
   
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    '...and now we\'re done!'

    st.title("THANK YOU :))")