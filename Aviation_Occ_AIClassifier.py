# Machine learning program for classifying aviation occurrences into Incident or Serious Incident

# Basic packages
import numpy as np
import pandas as pd
import streamlit as st 
 
# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Streamlit code
st.title('TSIB AI Occurrence Classifier')
st.write("""
# Occurrence classification using Machine Learning
Determines if an event is an "Incident" or "Serious Incident"
""")


# Dataset
X_predict = pd.DataFrame()
data = pd.DataFrame()
data = pd.read_csv("https://raw.githubusercontent.com/reccepython/Aviation_Occurrence_AIClassifier/main/CaseDatabase.csv")
data.drop('No', axis=1, inplace=True)
X = data.iloc[:,:-2].values
Y = data['Inc_SInc']



# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)


# Get user input parameters
st.sidebar.title('User Input Parameters:')
st.sidebar.title("""#
KEY: 0 - No, 1 - Yes""")

def get_user_input():
    Collision_NearCollision = st.sidebar.slider('Collision or near collision', 0,1)
    Nearly_CFIT = st.sidebar.slider('Near CFIT', 0,1)
    Takeoff_ClosedRunway = st.sidebar.slider('Take-off Closed Runway', 0,1)
    Takeoff_Taxiway	= st.sidebar.slider('Take-off Taxiway', 0,1)
    Landing_ClosedRunway = st.sidebar.slider('Landing Closed Runway', 0,1)	
    Fire = st.sidebar.slider('Fire', 0,1)
    Emergency_oxy = st.sidebar.slider('Emergency Oxygen Use', 0,1)
    Aircraft_System_Fail = st.sidebar.slider('Aircraft System Failure', 0,1)
    Incapacitation = st.sidebar.slider('Incapacitation', 0,1) 
    Incursion = st.sidebar.slider('Incursion', 0,1) 
    Fuel_related = st.sidebar.slider('Fuel Related', 0,1)
    Aircraft_Damage = st.sidebar.slider('Aircraft Damage', 0,1)

    user_data = {
        'Collision or near collision': Collision_NearCollision,
        'Near CFIT': Nearly_CFIT,
        'Take-off Closed Runway': Takeoff_ClosedRunway,
        'Take-off Taxiway': Takeoff_Taxiway,
        'Landing Closed Runway': Landing_ClosedRunway,
        'Fire': Fire,
        'Emergency Oxygen Use': Emergency_oxy,
        'Aircraft System Failure': Aircraft_System_Fail,
        'Incapacitation': Incapacitation,
        'Incursion': Incursion,
        'Fuel Related': Fuel_related,
        'Aircraft Damage': Aircraft_Damage
    }


# Transform user data into dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features


# Store user inputs at temporary location
user_input = get_user_input()

# For main page, use non-zero user inputs
user_input_mainpage = user_input.T[user_input.any()].T
user_input_mainpage = user_input_mainpage.transpose()


st.subheader('User Inputs:')
# st.write (user_input)
st.table (user_input_mainpage)



# Instantiate the Support Vector Classifier (SVC)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


svc = SVC(C=1.0, random_state=1, kernel='linear')
 

# Fit the model
svc.fit(X_train, y_train)


# Make the prediction
SVC_prediction = svc.predict(X_test)
print(accuracy_score(SVC_prediction, y_test))
print(confusion_matrix(SVC_prediction, y_test))


# Show model accuracy on app
accuracy = str(round(accuracy_score(SVC_prediction, y_test)*100, 1))

# Show the prediction
X_predict = sc.transform(user_input)
Y_predict = svc.predict(X_predict)

if user_input_mainpage.empty:
    Y_predict = str('No input has been provided yet.')

if Y_predict == 1:
    Y_predict = str('a Serious Incident')
elif Y_predict == 0 :
    Y_predict = str('an Incident')


st.subheader(f"""Classification (predicted up to {accuracy}% accuracy):
The occurrence is:""")
st.write(Y_predict)


# Streamlit code (ERC section)
st.title("""ICAO Serious Incident Classification Method
Use this method in conjunction with the TSIB AI Occurrence Classifier.
If the result between the AI Classifier and the ICAO method differs, take the more severe classification.""")

option1 = st.selectbox(
     'Was there a credible scenario by which this occurrence could have escalated to an "Accident"?',
     ('No', 'Yes'))

st.write('Could have escalated to an accident:', option1)

option2 = st.selectbox(
     'After assessing the remaining defences between this occurrence and the potential credible accident, was the defences "Effective - several defences prevented accident" or "Limited - few or no defences and accident only avoided due to luck"?',
     ('Effective', 'Limited'))

st.write('Remaining defences were:', option2)


if option1 == 'No':
    icao_ans = str('an Incident')
elif (option1 == 'Yes') and (option2 == 'Effective'):
    icao_ans = str('an Incident')
elif (option1 == 'Yes') and (option2 == 'Limited'):
    icao_ans = str('a Serious Incident')

st.subheader(f"""ICAO Classification Method:
The occurrence is:""")
st.write(icao_ans)

print(icao_ans)
print(Y_predict)


if (Y_predict != icao_ans) and (Y_predict == 'No input has been provided yet.'):
    st.subheader(""" """)
elif Y_predict != icao_ans:
    st.subheader("""**_There is a difference detected between the AI Classifier and the ICAO Classification Method_**""")

