import pandas as pd
df = pd.read_csv('predicted.csv')

def recommend(medicine):
    medicine =  medicine.lower()
    condition= df[df['drugName'] == medicine].condition
    condition.values[0]
    medicineName = df[df['condition'] == condition.values[0]]
    medicine_list = list(set(medicineName['drugName'].values[0:11]))
   
    return medicine_list


def recommendCondition(con):
    con = con.lower()
    conditionName = df[df['condition'] == con]
    med_list = list(set(conditionName['drugName'].values[0:11]))
   
    return med_list


def recommend_detail(medicine):
    medicine =  medicine.lower()
    result = df.loc[df['drugName'] == medicine]
    resultant_df = result.head(1).reset_index(drop= True)
    return resultant_df.to_dict()