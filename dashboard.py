import streamlit as st
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
q1 = pd.read_csv("output/q1.csv")
q2 = pd.read_csv("output/q2.csv")
q3 = pd.read_csv("output/q3.csv")
q4 = pd.read_csv("output/q4.csv")
q5 = pd.read_csv("output/q5.csv")

st.markdown('---')
st.write("# Big Data Project  \n Mady by Alexander Golovin and Ivan Kiselyev  \n Porto Seguro’s Safe Driver "
         "Prediction \n", "*Year*: **2023**")
st.markdown("""<style>body {
    background-color: #eee;
}

.fullScreenFrame > div {
    display: flex;
    justify-content: center;
}
</style>""", unsafe_allow_html=True)

st.image("https://storage.googleapis.com/kaggle-competitions/kaggle/7082/logos/header.png")

st.write("## Data description \n "
         "train.csv contains the training data, "
         "where each row corresponds to a policy holder,"
         " and the target columns signifies that a claim was filed.")

st.write(train.head(5))
st.write(train.describe())

st.write("test.csv contains the test data WITHOUT TARGET TO PREDICT.")
st.write(test.head(5))

st.markdown("---")
st.header("Exploratory Data Analysis")
st.subheader('Q1')
st.text('Query to find the average number of claims filed by each category in a categorical feature: This will help '
        'to understand if certain categories are more likely to file a claim.')
st.write(q1)

st.subheader('Q2')
st.write('Query to find the distribution of a continuous feature for each target class: It shows if the distribution of a feature is different for those who file a claim vs those who dont')
st.write(q2)

st.subheader('Q3')
st.write('Query to find the number of missing values for each feature, split by target class: it depicts whether missing data is more common in one class vs the other')
st.write(q3)

st.subheader('Q4')
st.write('This code calculates the correlation between binary classifications (for example here ps_ind_18_bin) binary feature and each of the other binary features in the train table. Firther we can do the same by adding additional corr functions for the remaining binary features to compute their correlations with ps_ind_18_bin and store the results in the correlation matrix. To To futher adjust the feature names with other column names.')
st.write(q4)

st.subheader('Q5')
st.write('Analyzing the distribution of values for selected calc features')
st.write(q5)





