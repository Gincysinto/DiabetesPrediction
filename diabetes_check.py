import tkinter as tk                       # GUI Toolkit
from tkinter import filedialog
from tkinter import *
import numpy as np
import lightgbm as lgb
import pandas as pd

def predictDiabetes():
    # Retrieve user input values from Entry widgets
    pregnanciesValue = float(pregnanciesEntry.get())
    glucoseValue = float(glucoseEntry.get())
    bloodPressureValue = float(bloodPressureEntry.get())
    skinThicknessValue = float(skinThicknessEntry.get())
    insulinValue = float(insulinEntry.get())
    BMIValue = float(BMIEntry.get())
    diabetesPedigreeFunctionValue = float(diabetesPedigreeFunctionEntry.get())
    ageValue = int(ageEntry.get())

    user_data = pd.DataFrame({'Pregnancies': [pregnanciesValue], 'Glucose' :[glucoseValue], 'BloodPressure': [bloodPressureValue], 'SkinThickness': [skinThicknessValue], 'Insulin': [insulinValue], 'BMI': [BMIValue], 'DiabetesPedigreeFunction': [diabetesPedigreeFunctionValue], 'Age': [ageValue], 'Glucose_CAT':[0], 'Life_Level_CAT':[0], 'Insulin_CAT_Prediabetes':[0], 'Insulin_CAT_Diabetes':[0], 'BloodPressure_CAT_Normal':[0], 'BloodPressure_CAT_Prehypertension':[0], 'BloodPressure_CAT_Hypertension':[0], 'BloodPressure_CAT_Hypertensive_Crisis':[0], 'BMI_CAT_Healthy':[0], 'BMI_CAT_Overweight':[0], 'BMI_CAT_Obese_Class1':[0], 'BMI_CAT_Obese_Class2':[0], 'BMI_CAT_Obese_Class3':[0], 'Age_CAT_Middle_Age_Adult':[0], 'Age_CAT_Senior_Adult':[0]})

    # user_data DataFrame 
    print("User Data:")
    print(user_data)
    
    # Schema synced new user_data DataFrame 
    print("synced User Data:")
    print(user_data)

    print("Head:")
    print(user_data.head)

    # Specify the path to the saved LGBM model file
    model_file_path='lgbm_Diabetes_classifier_model.txt'

    # Load the trained LGBM model while specifying categorical features
       
    lgbm_model = lgb.Booster(model_file=model_file_path)

    # Get the list of feature names from the model
    feature_names = lgbm_model.feature_name()

    # Print the feature names
    print("Feature Names:", feature_names)
    user_data = lgb.Dataset(data=user_data)
    # Convert the dataset to a LightGBM Dataset object if it's not already
    if not isinstance(user_data, lgb.Dataset):
        user_data = lgb.Dataset(data=user_data)

    # Make predictions using the model
    print("user_data.data:", user_data.data)
    predictions = int(np.round(lgbm_model.predict(user_data.data)))

    # Predictions 
    print("Predictions:")
    print(predictions)

    # Dictionary to label all traffic signs class.
    classes = {0:'No sign of Diabetics', 1:'Diabetes detected..' }

    value = classes[predictions]
    print(value)
    resultLabel.configure(foreground='#011638', text=value)

#initialise GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=600, relief='raised')
canvas1.pack() 

# Heading
label1 = tk.Label(root, text='Diabetes Prediction')
label1.config(font=('helvetica', 14))
root.title('Diabetes Prediction')
canvas1.create_window(200, 25, window=label1)

#Pregnancies
pregnanciesLabel = tk.Label(root, text='No of Pregnancies')
pregnanciesLabel.config(font=('helvetica', 10))
canvas1.create_window(200, 50, window=pregnanciesLabel)

pregnanciesEntry = tk.Entry(root) 
canvas1.create_window(200, 75, window=pregnanciesEntry)

#Glucose
glucoseLabel = tk.Label(root, text='Glucose')
glucoseLabel.config(font=('helvetica', 10))
canvas1.create_window(200, 100, window=glucoseLabel)

glucoseEntry = tk.Entry(root) 
canvas1.create_window(200, 125, window=glucoseEntry)

#BloodPressure
bloodPressureLabel = tk.Label(root, text='BloodPressure')
bloodPressureLabel.config(font=('helvetica', 10))
canvas1.create_window(200, 150, window=bloodPressureLabel)

bloodPressureEntry = tk.Entry(root) 
canvas1.create_window(200, 175, window=bloodPressureEntry)

#Skin Thickness
skinThicknessLabel = tk.Label(root, text='Skin Thickness')
skinThicknessLabel.config(font=('helvetica', 10))
canvas1.create_window(200, 200, window=skinThicknessLabel)

skinThicknessEntry = tk.Entry(root) 
canvas1.create_window(200, 225, window=skinThicknessEntry)

#Insulin
insulinLabel = tk.Label(root, text='Insulin')
insulinLabel.config(font=('helvetica', 10))
canvas1.create_window(200, 250, window=insulinLabel)

insulinEntry = tk.Entry(root) 
canvas1.create_window(200, 275, window=insulinEntry)

#BMI
BMILabel = tk.Label(root, text='BMI')
BMILabel.config(font=('helvetica', 10))
canvas1.create_window(200, 300, window=BMILabel)

BMIEntry = tk.Entry(root) 
canvas1.create_window(200, 325, window=BMIEntry)

#DiabetesPedigreeFunction
diabetesPedigreeFunctionLabel = tk.Label(root, text='Diabetes Pedigree Function')
diabetesPedigreeFunctionLabel.config(font=('helvetica', 10))
canvas1.create_window(200, 350, window=diabetesPedigreeFunctionLabel)

diabetesPedigreeFunctionEntry = tk.Entry(root) 
canvas1.create_window(200, 375, window=diabetesPedigreeFunctionEntry)

#Age
ageLabel = tk.Label(root, text='Age')
ageLabel.config(font=('helvetica', 10))
canvas1.create_window(200, 400, window=ageLabel)

ageEntry = tk.Entry(root) 
canvas1.create_window(200, 425, window=ageEntry)

#Butto for prediction
button1 = tk.Button(text='Predict',  bg='brown', fg='black',command=predictDiabetes, font=('helvetica', 9, 'bold'))
canvas1.create_window(200, 475, window=button1)

#result label
resultLabel = tk.Label(root)
resultLabel.config(font=('Cambria', 20,'bold'))
canvas1.create_window(200, 500, window=resultLabel)
resultLabel.pack(expand=True)


root.mainloop()