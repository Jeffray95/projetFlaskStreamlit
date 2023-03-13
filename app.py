#import libraries
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt; 
import seaborn as sns #graphiuqe
import streamlit as st #view
from sklearn.datasets import load_iris
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)

#data transformation
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, Binarizer

#algorithm of classification
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#Algorith of regression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#load data
dataset= load_iris()
print(dataset)
#create dataframe with iris data
data=dataset.data #pour .data pour les input et .target pour les output
target_names=dataset.target_names#class
feature_names=dataset.feature_names #Colonne
df=pd.DataFrame(data, columns=feature_names)
#import fumtion EDA
from eda import eda
#streamlit
st.set_page_config(page_title="EDA and ML Dashbord",
                    layout="centered",
                    initial_sidebar_state="auto")
#add title and markdown description
            
st.title("EDA AND Predictive modeling dashbord")

#define sidebar and sidebar option
options=["EDA","predictive modeling"]
selected_option=st.sidebar.selectbox("Select options",options)
#EDA
if selected_option=="EDA":
    #call function from eda.py
    eda(df,target_names, feature_names)    
#predictive modeling
elif selected_option=="predictive modeling":
    st.subheader("predictive modelling")
    st.write("Choose a transform type an model from the option below")
    X=df.values
    Y=dataset.target
    st.write(X)
    test_proportion=0.30
    seed=5
    X_train, X_test, Y_train,Y_test=train_test_split(X, Y, test_size=test_proportion, random_state=seed)
    transform_options=["none",
                       "StandarScaler",
                       "Normalizer",
                       "MinMaxScaler"]
    transform=st.selectbox("Select data transform",transform_options)
    if transform == "StandarScaler":
        scaler=StandarScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
    elif transform == "Normalizer":
        scaler=Normalizer()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
    elif transform == "MinMaxScaler":
        scaler=MinMaxScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
    else:
        X_train=X_train
        X_test=X_test
    classifier_list  = ["LogisticRegression",
                        "SVM",
                        "DecisionTree",
                        "KNeighbors",
                        "RandomForest"]
    classifier=st.selectbox("Select classifier",classifier_list)
    if classifier == "LogisticRegression":
        st.write("here are the result of LogisticRegression model")
        solver_value =st.selectbox("select solver",
                        ["lbfgs", 
                        "liblinear",
                        "newton-cg",
                        "newton cholesky"])
        model=LogisticRegression(solver=solver_value)
        model.fit(X_train,Y_train)
        #   make prediction
        y_pred=model.predict(Y_test)
        accuracy=accuracy_score(Y_test,y_pred)
        precision=precision_score(Y_test,y_pred,average="micro")
        recall=recall_score(Y_test,y_pred,average="macro")
        f1 =f1_score(Y_test,y_pred,average="weighted")
        # display result
        st.write(f"Accuracy : ", {accuracy})
        st.write(f"Precision : ",{precision})
        st.write(f"Recall : ",{recall})
        st.write(f"F1-Score: ",{f1_score})
        st.write(confusion_matrix(Y_test, y_pred))
    elif classifier == "DecisionTree":
        st.write("here are the result of DecisionTree model")            
        model=DecisionTreeClassifier()
        model.fit(X_train,Y_train)
            #make prediction
        y_pred=model.predict(Y_test)
        accuracy=accuracy_score(Y_test,y_pred)
        precision=precision_score(Y_test,y_pred,average="micro")
        recall=recall_score(Y_test,y_pred,average="macro")
        f1 =f1_score(Y_test,y_pred,average="weighted")
            # display result
        st.write(f"Accuracy : ", {accuracy})
        st.write(f"Precision : ",{precision})
        st.write(f"Recall : ", {recall})
        st.write(f"F1-Score : ", {f1_score})
        st.write(confusion_matrix(Y_test, y_pred))
        
                        



