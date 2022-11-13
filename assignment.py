import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas_profiling import ProfileReport

# web app ka title

st.markdown=(''' 
# **Exploratory Data Analysis web Application**
This app is developed by Fria Khizar undersupervision of Baba Aammar called **EDA App**
 ''')

# uploading a file from pc

with st.sidebar.header(" Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type= ['csv'])
    df = pd.read_csv('survey_lung_cancer.csv')
    
st.markdown=''' Pandas Profiling'''
# profiling report for pandas
if uploaded_file is not None:
    @st.cache   # for increasing the loading  speed of data
    
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file, upload kr bhi do ab ya kaam nhi lena?')
    if st.button('Press to use example data'):
        # example dataset
        
        def load_data():
            a = pd.DataFrame( np.random.rand(100,5))
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DF**')
        st.write(df)
        st.write('---')
        st.header('**Profiling report with pandas**')
        st_profile_report(pr)
    

# make containers
header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Lung cancer ki wja r Prediction")
    st.text("This project is about the predicted reasons of lung cancer ")
with data_sets :
    st.header("Lung Cancer ki wajuhaat")
    st.text("We will work with Lung Cancer dataset")
    # Impoer Data
    df = pd.read_csv('survey_lung_cancer.csv')
    df = df.dropna()
    st.write(df.describe())
    
    st.subheader("Omar k hisaab se farq")
    st.bar_chart(df['AGE'].value_counts())
    
    # Other Plot
    st.subheader("Smoking k hisaab se farq")
    st.bar_chart(df['SMOKING'].value_counts())
    # Barplot
    st.bar_chart(df['ALCOHOL CONSUMING'].sample(50))  # or head(10)
    
    
    st.bar_chart(df['COUGHING'].value_counts())
    

    
    
    
    
with model_training:
    st.header("Lung cancer ka kia bna?_ Model Training")
    st.text("is mn hm apny parameters ko km ya ziada kren gn")
    # making columns
    input, display = st.columns(2)
    
    # pehly column main ap k selection poits hain
    max_depth = input.slider("How many products do you know?", min_value=0, max_value=100,value= 20, step=5)
    
# n_estimators
n_estimators = input.selectbox("How many tree should be there in a RF?", options=[50,100,200,300, 'No Limit'])


# adding list of features
input.write(df.columns)



#  input features from user
input_features = input.text_input('Which feature we should use?')


# Machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# yahan pr hm aik condition lagayen gn
if n_estimators == 'No limit' :
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth = max_depth, n_estimators=n_estimators)


# Define x and y
X = df[[input_features]]
y = df[["ALCOHOL CONSUMING"]]

# fit our model
model.fit(X,y)
pred = model.predict(y)


# Display metrices
display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean absolute error of the model is: ")
display.write(mean_squared_error(y, pred))
display.subheader("Mean absolute error of the model is: ")
display.write(r2_score(y, pred))





