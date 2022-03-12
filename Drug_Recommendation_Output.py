import pandas as pd
df = pd.read_csv('predicted.csv')

def recommend(medicine):
    medicine =  medicine.lower()
    condition= df[df['drugName'] == medicine].condition
    condition.values[0]
    medicineName = df[df['condition'] == condition.values[0]]
    medicine_list = list(medicineName['drugName'].values[0:6])
   
    return medicine_list


