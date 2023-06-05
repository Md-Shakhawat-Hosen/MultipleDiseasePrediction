# -*- coding: utf-8 -*-


import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification
from tumor_img import teachable_machine_classification2


# loading the saved models

diabetes_model = pickle.load(open('ML_model/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load( open('ML_model/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('ML_model/parkinsons_model.sav', 'rb'))

covid_model = pickle.load(open('ML_model/trained_covid_model.sav', 'rb'))

blood_model = pickle.load(open('ML_model/trained_model_hbp.sav', 'rb'))

lung_model = pickle.load(open('ML_model/trained_model_lung_cancer.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Covid-19 Prediction',
                           'High BP Prediction',
                           'Lung Cancer Prediction',
                           'Breast Cancer Prediction',
                           'Brain Tumor Prediction',
                           'Blood Group Detection',
                           'Pneumonia Detection',
                           'Mask Detection Capture Image',
                           'Mask Detection Upload Image'],
                           icons=['activity', 'heart', 'person', 'circle', 'activity', 'heart-fill', 'gender-female', "people-fill", "people-fill", "people-fill", 'mask', 'mask'],
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
    diabetes_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diabetes_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diabetes_prediction[0] == 1):
          diabetes_diagnosis = 'The person is diabetic'
        else:
          diabetes_diagnosis = 'The person is not diabetic'
        
    st.success(diabetes_diagnosis)




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
        covid_prediction = covid_model.predict(
            [[idVal,Pregnancies, Glucose, BloodPressure]])

        if (covid_prediction[0] == 1):
          covid_diagnosis = 'The person Covid-19 Positive'
        else:
          covid_diagnosis = 'The person Covid-19 Negative'

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
        blood_prediction = blood_model.predict(
            [[valId, pseudo_psu, pseudo_stratum, stat_weight, age, sex, race, body_weight, height, avg_systolic_bp, avg_diastolic_bp, smoked_alot, currently_smokes, smoking, serum_cholesterol]])

        if (blood_prediction[0] == 1):
          blood_diagnosis = 'The person has no High Blood Pressure'
        else:
          blood_diagnosis = 'The person has High Blood Pressure'

    st.success(blood_diagnosis)

#Lung Cancer Prediction Page:

if(selected == "Lung Cancer Prediction"):

    #page title
    st.title("Lung Cancer Prediction using Machine Learning")


# getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        GENDER = st.text_input("GENDER")

    with col2:
        AGE = st.text_input("AGE")

    with col3:
        SMOKING = st.text_input("SMOKING")

    with col4:
        YELLOW_FINGERS = st.text_input("YELLOW_FINGERS")

    with col1:
        ANXIETY = st.text_input("ANXIETY")

    with col2:
        PEER_PRESSURE = st.text_input("PEER_PRESSURE")

    with col3:
        CHRONIC_DISEASE = st.text_input("CHRONIC DISEASE")

    with col4:
        FATIGUE = st.text_input("FATIGUE")

    with col1:
        ALLERGY = st.text_input("ALLERGY")

    with col2:
        WHEEZING = st.text_input("WHEEZING")

    with col3:
        ALCOHOL_CONSUMING = st.text_input("ALCOHOL CONSUMING")

    with col4:
        COUGHING = st.text_input("COUGHING")

    with col1:
        SHORTNESS_OF_BREATH = st.text_input("SHORTNESS OF BREATH")

    with col2:
        SWALLOWING_DIFFICULTY = st.text_input("SWALLOWING DIFFICULTY")

    with col3:
        CHEST_PAIN = st.text_input("CHEST PAIN")


# code for Prediction
    lung_cancer_result = " "

    # creating a button for Prediction

    if st.button("Lung Cancer Test Result"):
        lung_cancer_report = lung_model.predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
                                                 FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])

        if (lung_cancer_report[0] == 0):
          lung_cancer_result = "Hurrah! You have no Lung Cancer."
        else:
          lung_cancer_result = "Sorry! You have Lung Cancer."

    st.success(lung_cancer_result)

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


#Blood Group Detection
if (selected == 'Blood Group Detection'):
    st.title("Image Classification")
    st.header("Blood Group Classification")
    st.text("Upload a scan for Classification")

    #uploaded_file = st.camera_input("Choose a scan ...", key="firstCamera")
    uploaded_file = st.file_uploader("Choose a scan ...", type="jpg")

    label_diagnosis = ''
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Scan.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label_diagnosis = teachable_machine_classification2(
            image, 'model/keras_model_blood_group.h5')
        if label_diagnosis == 0:
            label_diagnosis = 'A Positive'
        elif label_diagnosis == 1:
            label_diagnosis = 'A Negative'
        elif label_diagnosis == 2:
            label_diagnosis = 'B Positive'
        elif label_diagnosis == 3:
            label_diagnosis = 'B Negative'
        elif label_diagnosis == 4:
            label_diagnosis = 'AB Positive'
        elif label_diagnosis == 5:
            label_diagnosis = 'AB Negative'
        elif label_diagnosis == 6:
            label_diagnosis = 'O Positive'
        elif label_diagnosis == 7:
            label_diagnosis = 'O Negative'

    st.success(label_diagnosis)


#Pneumonia Detection
if (selected == 'Pneumonia Detection'):
    st.title("Image Classification")
    st.header("Pneumonia image Classification")
    st.text("Upload a scan for Classification")

    #uploaded_file = st.camera_input("Choose a scan ...", key="firstCamera")
    uploaded_file = st.file_uploader("Choose a scan ...", type="jpeg")

    label_diagnosis = ''
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Scan.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label_diagnosis = teachable_machine_classification2(
            image, 'model/keras_model_pneumonia.h5')
        if label_diagnosis == 0:
            label_diagnosis = 'Normal'
        elif label_diagnosis == 1:
            label_diagnosis = 'Pneumonia'
    st.success(label_diagnosis)

#Mask Detection
if (selected == 'Mask Detection Capture Image'):
    st.title("Image Classification")
    st.header("Mask Detection")
    st.text("Capture a Image for Classification")
    
    uploaded_file = st.camera_input("Capture Image...", key="firstCamera")
    #uploaded_file = st.file_uploader("Choose a scan ...", type="jpg")
    
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

#Mask Detection upload image
if (selected == 'Mask Detection Upload Image'):
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
        label_diagnosis = teachable_machine_classification(
            image, 'model/keras_model_mask.h5')
        if label_diagnosis == 0:
            #st.write("With Mask")
            label_diagnosis = 'The person with Mask'
        else:
            #st.write("Without Mask")
            label_diagnosis = 'The person has no Mask'

    st.success(label_diagnosis)
    










