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
data = pd.read_excel("https://raw.githubusercontent.com/reccepython/Aviation_Occurrence_AIClassifier/main/CaseDatabase.xlsx", engine='openpyxl')
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

    Pushback = st.sidebar.slider('Pushback Phase', 0,1)	
    TaxiingTowing = st.sidebar.slider('Taxiing or Towing', 0,1)	
    Takeoff = st.sidebar.slider('Take-off or TOGA', 0,1)	
    InflightPhase = st.sidebar.slider('During Inflight Phase', 0,1)	
    Landing = st.sidebar.slider('Landing Phase', 0,1)	
    Parking = st.sidebar.slider('Parking or Parked Phase', 0,1)	
    Groundhandling_Maintenance = st.sidebar.slider('Maintenance Related During Groundhandling', 0,1)	
    Groundhandling_Other = st.sidebar.slider('Other Groundhandling ', 0,1)	
    AircraftLoadRelated = st.sidebar.slider('Issues With Aircraft Load Or Its Planning', 0,1)	
    OEM_MRO_Related = st.sidebar.slider('OEM or MRO Related', 0,1)	
    #Runway_NormalOps = st.sidebar.slider('Runway Under Normal Ops', 0,1)	
    Runway_SpecialOps = st.sidebar.slider('Runway Under Special Ops', 0,1)	
    Taxiway = st.sidebar.slider('Taxiway', 0,1)	
    Collision = st.sidebar.slider('Collision', 0,1)	
    Near_Collision = st.sidebar.slider('Near Collision', 0,1)	
    Incursion = st.sidebar.slider('Incursion', 0,1)	
    Excursion = st.sidebar.slider('Excursion', 0,1)	
    RunwayOverrun = st.sidebar.slider('Runway Overrun', 0,1)	
    Turbulence = st.sidebar.slider('Turbulence', 0,1)	
    Weather = st.sidebar.slider('Weather', 0,1)	
    Aerodrome = st.sidebar.slider('Aerodrome', 0,1)	
    ATC_Resource = st.sidebar.slider('ATC Resource Issues', 0,1)	
    ATC_Comms = st.sidebar.slider('ATC Communications Issues', 0,1)	
    ATC_Actions = st.sidebar.slider('ATC Actions', 0,1)	
    LossOfSeparation = st.sidebar.slider('Loss Of Separation', 0,1)	
    TCAS_RA = st.sidebar.slider('TCAS Resolution Advisory', 0,1)	
    TCAS_TA = st.sidebar.slider('TCAS Traffic Advisory', 0,1)	
    EGPWS = st.sidebar.slider('EGPWS', 0,1)	
    Takeoff_Performance = st.sidebar.slider('Take-off Performance', 0,1)	
    Flight_Performance = st.sidebar.slider('Flight Performance', 0,1)	
    InflightLossOfControl_Stall = st.sidebar.slider('Loss Of Control During Flight ', 0,1)	
    MinSafeAltitude = st.sidebar.slider('Minimal Safe Altitude', 0,1)	
    LandingConfig = st.sidebar.slider('Landing Configuration Issues', 0,1)	
    Crew_Resource = st.sidebar.slider('Flight Crew Resourse Issues', 0,1)	
    Crew_Comms = st.sidebar.slider('Flight Crew Communications Issues', 0,1)	
    Crew_Actions = st.sidebar.slider('Flight Crew Actions', 0,1)	
    InputError = st.sidebar.slider('Input Error or Omission', 0,1)	
    AircraftDamage_Replaceable = st.sidebar.slider('Aircraft Damage Replaceable', 0,1)	
    AircraftDamage_MinorRepair = st.sidebar.slider('Aircraft Damage Minor Repair', 0,1)	
    AircraftDamage_MajorRepair = st.sidebar.slider('Aircraft Damage Major Repair', 0,1)	
    TailStrike = st.sidebar.slider('Tailstrike', 0,1)	
    FOD = st.sidebar.slider('FOD', 0,1)	
    Birdstrike = st.sidebar.slider('Birdstrike', 0,1)	
    LandingGears = st.sidebar.slider('Landing Gears', 0,1)	
    HydraulicSystem = st.sidebar.slider('Hydraulic System', 0,1)	
    FuelSystem = st.sidebar.slider('Fuel System', 0,1)	
    ElectricalSystem = st.sidebar.slider('Electrical System', 0,1)	
    FlightControlSystem = st.sidebar.slider('Flight Control System', 0,1)	
    ElectronicAvionicsRelated = st.sidebar.slider('Electronic or Avionics Related', 0,1)	
    Engine_Failure = st.sidebar.slider('Engine Failure or Unusable', 0,1)	
    Engine_Issues = st.sidebar.slider('Engine Issues (Other)', 0,1)	
    Engine_Damage = st.sidebar.slider('Engine Damage', 0,1)	
    Parts_Liberated = st.sidebar.slider('Parts Liberated', 0,1)	
    Smoke_OdourRelated = st.sidebar.slider('Smoke Fumes Odour Related', 0,1)	
    Fire_Indication = st.sidebar.slider('Fire Indication Alerts', 0,1)	
    FirePersist = st.sidebar.slider('Fire Indication Persist Despite Actions', 0,1)	
    Fire_Engines = st.sidebar.slider('Fire (Engines)', 0,1)	
    Fire_Others = st.sidebar.slider('Fire (Others)', 0,1)	
    PressurisationRelated = st.sidebar.slider('Pressurisation Related', 0,1)	
    Emergency_oxy = st.sidebar.slider('Emergency Oxygen Use', 0,1)	
    Incapacitation = st.sidebar.slider('Incapacitation', 0,1)	
    Injuries = st.sidebar.slider('Injuries', 0,1)
       

    user_data = {
        'Pushback Phase' : Pushback,	
        'Taxiing or Towing' : TaxiingTowing,	
        'Take-off or TOGA' : Takeoff,	
        'During Inflight Phase' : InflightPhase,	
        'Landing Phase' : Landing,	
        'Parking or Parked Phase' : Parking,	
        'Maintenance Related During Groundhandling' : Groundhandling_Maintenance,	
        'Other Groundhandling' : Groundhandling_Other,
        'Issues With Aircraft Load Or Its Planning' : AircraftLoadRelated,	
        'OEM or MRO Related' : OEM_MRO_Related,	
        #'Runway Under Normal Ops' : Runway_NormalOps,	
        'Runway Under Special Ops' : Runway_SpecialOps,	
        'Taxiway' : Taxiway,	
        'Collision' : Collision,	
        'Near Collision' : Near_Collision,	
        'Incursion' : Incursion,	
        'Excursion' : Excursion,	
        'Runway Overrun' : RunwayOverrun,	
        'Turbulence' : Turbulence,	
        'Weather' : Weather,	
        'Aerodrome' : Aerodrome,	
        'ATC Resource Issues' : ATC_Resource,	
        'ATC Communications Issues' : ATC_Comms,	
        'ATC Actions' : ATC_Actions,	
        'Loss Of Separation' : LossOfSeparation,	
        'TCAS Resolution Advisory' : TCAS_RA,	
        'TCAS Traffic Advisory' : TCAS_TA,	
        'EGPWS' : EGPWS,	
        'Take-off Performance' : Takeoff_Performance,	
        'Flight Performance' : Flight_Performance,	
        'Loss Of Control During Flight' : InflightLossOfControl_Stall,	
        'Minimal Safe Altitude' : MinSafeAltitude, 
        'Landing Configuration Issues' : LandingConfig, 
        'Flight Crew Resourse Issues' : Crew_Resource, 
        'Flight Crew Communications Issues' : Crew_Comms, 
        'Flight Crew Actions' : Crew_Actions, 
        'Input Error or Omission' : InputError, 
        'Aircraft Damage Replaceable' : AircraftDamage_Replaceable, 
        'Aircraft Damage Minor Repair' : AircraftDamage_MinorRepair, 
        'Aircraft Damage Major Repair' : AircraftDamage_MajorRepair, 
        'TailStrike' : TailStrike,	
        'FOD' : FOD,	
        'Birdstrike' : Birdstrike,	
        'Landing Gears' : LandingGears,	
        'Hydraulic System' : HydraulicSystem,	
        'Fuel System' : FuelSystem,	
        'Electrical System' : ElectricalSystem,	
        'Flight Control System' : FlightControlSystem,	
        'Electronic or Avionics Related' : ElectronicAvionicsRelated,	
        'Engine Failure or Unusable' : Engine_Failure,	
        'Engine Issues (Other)' : Engine_Issues,	
        'Engine Damage' : Engine_Damage,	
        'Parts Liberated' : Parts_Liberated,	
        'Smoke Fumes Odour Related' : Smoke_OdourRelated,	
        'Fire Indication Alerts' : Fire_Indication,
        'Fire Indication Persist Despite Actions' : FirePersist,
        'Fire (Engines)' : Fire_Engines,
        'Fire (Others)' : Fire_Others,
        'Pressurisation Related' : PressurisationRelated,	
        'Emergency Oxygen Use' : Emergency_oxy,	
        'Incapacitation' : Incapacitation,	
        'Injuries' : Injuries,
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

# End of code