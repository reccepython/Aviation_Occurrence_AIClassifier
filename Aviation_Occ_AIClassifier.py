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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2, stratify=Y)


# Get user input parameters
st.sidebar.title('User Input Parameters:')
st.sidebar.title("""#
Check all that applies:""")

def get_user_input():

    with st.sidebar.expander("Phase of flight"):
        Pushback = st.checkbox('Pushback Phase')	
        TaxiingTowing = st.checkbox('Taxiing or Towing')	
        Takeoff = st.checkbox('Take-off or TOGA')	
        InflightPhase = st.checkbox('During Inflight Phase')	
        Landing = st.checkbox('Landing Phase')	
        Parking = st.checkbox('Parking or Parked Phase')	
    with st.sidebar.expander("Pre-flight related"):
        Groundhandling_Maintenance = st.checkbox('Maintenance Related During Groundhandling')	
        Groundhandling_Other = st.checkbox('Other Groundhandling ')	
        AircraftLoadRelated = st.checkbox('Issues With Aircraft Load Or Its Planning')
    with st.sidebar.expander("OEM or MRO related"):	
        OEM_MRO_Related = st.checkbox('OEM or MRO Related')	
        #Runway_NormalOps = st.sidebar.checkbox('Runway Under Normal Ops')	
    with st.sidebar.expander("Aerodrome related"):
        Runway_SpecialOps = st.checkbox('Runway Under Special Ops')	
        Taxiway = st.checkbox('Taxiway')	
        Incursion = st.checkbox('Incursion')	
        Excursion = st.checkbox('Excursion')	
        RunwayOverrun = st.checkbox('Runway Overrun')
        Aerodrome = st.checkbox('Aerodrome')	
    with st.sidebar.expander("Weather related"):	
        Turbulence = st.checkbox('Turbulence')	
        Weather = st.checkbox('Weather')	
    with st.sidebar.expander("ATC related"):
        ATC_Resource = st.checkbox('ATC Resource Issues')	
        ATC_Comms = st.checkbox('ATC Communications Issues')	
        ATC_Actions = st.checkbox('ATC Actions')
    with st.sidebar.expander("Separation related"):	
        LossOfSeparation = st.checkbox('Loss Of Separation')	
        TCAS_RA = st.checkbox('TCAS Resolution Advisory')	
        TCAS_TA = st.checkbox('TCAS Traffic Advisory')	
        EGPWS = st.checkbox('EGPWS')	
    with st.sidebar.expander("Flight performance related"):
        Takeoff_Performance = st.checkbox('Take-off Performance')	
        Flight_Performance = st.checkbox('Flight Performance')	
        InflightLossOfControl_Stall = st.checkbox('Loss Of Control During Flight ')	
        MinSafeAltitude = st.checkbox('Minimal Safe Altitude')	
        LandingConfig = st.checkbox('Landing Configuration Or Performance Issues')
    with st.sidebar.expander("Flight crew related"):	
        Crew_Resource = st.checkbox('Flight Crew Resourse Issues')	
        Crew_Comms = st.checkbox('Flight Crew Communications Issues')	
        Crew_Actions = st.checkbox('Flight Crew Actions')	
    with st.sidebar.expander("Input error or Omission by any party"):
        InputError = st.checkbox('Input Error or Omission')	
    with st.sidebar.expander("Aircraft damage assessment"):
        AircraftDamage_Replaceable = st.checkbox('Aircraft Damage Replaceable')	
        AircraftDamage_MinorRepair = st.checkbox('Aircraft Damage Minor Repair')	
        AircraftDamage_MajorRepair = st.checkbox('Aircraft Damage Major Repair')	
        TailStrike = st.checkbox('Tailstrike')	
        FOD = st.checkbox('FOD')	
        Birdstrike = st.checkbox('Birdstrike')
        Collision = st.checkbox('Collision')	
        Near_Collision = st.checkbox('Near Collision')	
    with st.sidebar.expander("Aircraft system related"):	
        LandingGears = st.checkbox('Landing Gears')	
        HydraulicSystem = st.checkbox('Hydraulic System')	
        FuelSystem = st.checkbox('Fuel System')	
        ElectricalSystem = st.checkbox('Electrical System')	
        FlightControlSystem = st.checkbox('Flight Control System')	
        ElectronicAvionicsRelated = st.checkbox('Electronic or Avionics Related')
    with st.sidebar.expander("Engine issues"):	
        Engine_Failure = st.checkbox('Engine Failure or Unusable')	
        Engine_Issues = st.checkbox('Engine Issues (Other)')	
        Engine_Damage = st.checkbox('Engine Damage')	
    with st.sidebar.expander("Parts liberated from aircraft/engine"):
        Parts_Liberated = st.checkbox('Parts Liberated')
    with st.sidebar.expander("Fire/Smoke/Odour related"):	
        Smoke_OdourRelated = st.checkbox('Smoke Fumes Odour Related')	
        Fire_Indication = st.checkbox('Fire Indication Alerts')	
        FirePersist = st.checkbox('Fire Indication Persist Despite Actions')	
        Fire_Engines = st.checkbox('Fire (Engines)')	
        Fire_Others = st.checkbox('Fire (Others)')
    with st.sidebar.expander("Pressurisation related issues"):	
        PressurisationRelated = st.checkbox('Pressurisation Related')	
        Emergency_oxy = st.checkbox('Emergency Oxygen Use')	
    with st.sidebar.expander("Incapacitation/Injuries"):
        Incapacitation = st.checkbox('Incapacitation')	
        Injuries = st.checkbox('Injuries')
        

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
            'Incursion' : Incursion,	
            'Excursion' : Excursion,	
            'Runway Overrun' : RunwayOverrun,	
            'Aerodrome' : Aerodrome,	
            'Turbulence' : Turbulence,	
            'Weather' : Weather,	
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
            'Landing Configuration Or Performance Issues' : LandingConfig, 
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
            'Collision' : Collision,	
            'Near Collision' : Near_Collision,	
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
    features = pd.DataFrame(user_data, index = ["Parameter entered"])
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


st.subheader(f"""AI Classification Result:
The occurrence is:""")
st.write(Y_predict)


print(Y_predict + '\n')

# End of code