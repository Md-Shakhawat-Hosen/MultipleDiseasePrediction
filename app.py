# -*- coding: utf-8 -*-


import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification


# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

covid_model = pickle.load(open('trained_covid_model.sav','rb'))

blood_model = pickle.load(open('trained_model_hbp.sav', 'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Covid-19 Prediction',
                           'High BP Prediction',
                           'Breast Cancer Prediction',
                           'Brain Tumor Prediction',
                           'Mask Detection'],
                          icons=['activity','heart','person','list','list','list','list','mask'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)


# Covid-19 Prediction Page
if (selected == 'Covid-19 Prediction'):

    # page title
    st.title('Covid-19 Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        idVal = st.text_input('Serial Number')

    with col2:
        Pregnancies = st.text_input('Oxygen')

    with col3:
        Glucose = st.text_input('Pulse Rate')

    with col1:
        BloodPressure = st.text_input('Temperature')


    # code for Prediction
    covid_diagnosis = ''

    # creating a button for Prediction

    if st.button('COVID-19 Test Result'):
        diab_prediction = covid_model.predict(
            [[idVal,Pregnancies, Glucose, BloodPressure]])

        if (diab_prediction[0] == 1):
          covid_diagnosis = 'The person COVID positive'
        else:
          covid_diagnosis = 'The person COVID negative'

    st.success(covid_diagnosis)



    # High Blood Prediction Page
if (selected == 'High BP Prediction'):

    # page title
    st.title('High Blood Pressure Prediction using ML')

    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        valId = st.text_input('Serial Number')

    with col2:
        pseudo_psu = st.text_input('Pseudo_psu')

    with col3:
        pseudo_stratum = st.text_input('Pseudo_stratum')

    with col4:
        stat_weight = st.text_input('Stat_Weight')

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('SEX')

    with col3:
        race = st.text_input('Race')

    with col4:
        body_weight = st.text_input('Body_Weight')

    with col1:
        height = st.text_input('Height')

    with col2:
        avg_systolic_bp = st.text_input('Avg_systolic_bp')

    with col3:
        avg_diastolic_bp = st.text_input('Avg_diastolic_bp')

    with col4:
        smoked_alot = st.text_input('Smoked_alot')
    
    with col1:
        currently_smokes = st.text_input('Currently_smokes')

    with col2:
        smoking = st.text_input('Smoking')

    with col3:
        serum_cholesterol = st.text_input('Serum_cholesterol')

    

    # code for Prediction
    blood_diagnosis = ''

    # creating a button for Prediction

    if st.button('High Blood Pressure Test Result'):
        diab_prediction = blood_model.predict(
            [[valId, pseudo_psu, pseudo_stratum, stat_weight, age, sex, race, body_weight, height, avg_systolic_bp, avg_diastolic_bp, smoked_alot, currently_smokes, smoking, serum_cholesterol]])

        if (diab_prediction[0] == 1):
          blood_diagnosis = 'The person has no High Blood Pressure'
        else:
          blood_diagnosis = 'The person has High Blood Pressure'

    st.success(blood_diagnosis)

# Breast Cancer
if (selected == 'Breast Cancer Prediction'):
    st.title("Image Classification")
    st.header("Breast Cancer Ultrasound Classification")
    st.text("Upload a scan for Classification")


    uploaded_file = st.file_uploader("Choose a scan ...", type="png")

    label_diagnosis = ''
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Scan.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label_diagnosis = teachable_machine_classification(image, 'model/keras_model.h5')
        if label_diagnosis == 0:
            #st.write("The scan is normal")
            label_diagnosis = 'The scan is normal'
        elif label_diagnosis == 1:
            #st.write("The scan is malignant")
            label_diagnosis = 'The scan is malignant'
        elif label_diagnosis == 2:
            #st.write("The scan is benign")
            label_diagnosis = 'The scan is benign'

    st.success(label_diagnosis)

#Brain Tumor
if (selected == 'Brain Tumor Prediction'):
    st.title("Image Classification")
    st.header("Brain Tumor Classification")
    st.text("Upload a scan for Classification")

    #uploaded_file = st.camera_input("Choose a scan ...", key="firstCamera")
    uploaded_file = st.file_uploader("Choose a scan ...", type="jpg")
    
    label_diagnosis = ''
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Scan.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label_diagnosis = teachable_machine_classification(image, 'model/keras_model_tumor.h5')
        if label_diagnosis == 0:
            #st.write("Pituitary Tumor")
            label_diagnosis = 'Pituitary Tumor'
        else:
            #st.write("No Tumor")
            label_diagnosis = 'No Tumor'
    
    st.success(label_diagnosis)


#Mask Detection
if (selected == 'Mask Detection'):
    st.title("Image Classification")
    st.header("Mask Detection")
    st.text("Capture a Image for Classification")
    
    #uploaded_file = st.camera_input("Capture Image...", key="firstCamera")
    uploaded_file = st.file_uploader("Choose a scan ...", type="jpg")
    
    label_diagnosis = ''
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Capture Image', use_column_width=True)
        st.write("")
        st.write("Detecting...")
        label_diagnosis = teachable_machine_classification(image, 'model/keras_model_mask.h5')
        if label_diagnosis == 0:
            #st.write("With Mask")
            label_diagnosis = 'The person with Mask'
        else:
            #st.write("Without Mask")
            label_diagnosis = 'The person has no Mask'
    
    st.success(label_diagnosis)
    










