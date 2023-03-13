
#import libraries
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt; 
import seaborn as sns #graphiuqe
import streamlit as st #view
from sklearn.datasets import load_iris
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)

def eda(df,target_names, feature_names):    
    st.subheader("Exploratory data analisis and visualization")
    st.write("Choose a plot type from the option below :")
#add option to show/hide data
    if st.checkbox("show raw data"):
        st.write(df)
    #add option to show missing value isna.sum()
    if st.checkbox("show missing value"):
        st.write(df.isna().sum())
    if st.checkbox("data type"):
        st.write(df.dtypes)
    if st.checkbox(" show Descriptive data"):
        st.write(df.describe())
    if st.checkbox("Show corelation matrix"):
        corr=df.corr()
        mask = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr,mask=mask, annot=True, cmap="coolwarm")
        st.pyplot()
    if st.checkbox("Show histogramme for each attribute"):
        for col in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df[col],bins=20, density=True, alpha=0.5)
            ax.set_title(col)
            st.pyplot(fig)
    if st.checkbox("Show density for each attribute"):
        for col in df.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], fill=True) 
            ax.set_title(col)
            st.pyplot(fig)
    if st.checkbox("Show Scatter for each attribute"):
        fig=px.scatter(df, x = feature_names[0], y = feature_names[1])
        st.plotly_chart(fig)
