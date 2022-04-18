import pandas as pd
df = pd.read_csv('predicted.csv')

def recommend(medicine):
    medicine =  medicine.lower()
    condition= df[df['drugName'] == medicine].condition
    condition.values[0]
    medicineName = df[df['condition'] == condition.values[0]]
    medicine_list = list(medicineName['drugName'].values[0:6])
   
    return medicine_list


def recommendCondition(con):
    con = con.lower()
    conditionName = df[df['condition'] == con]
    medicines_list = list(conditionName['drugName'].values[0:6])
   
    return medicines_list


def recommend_detail(medicine):
    medicine =  medicine.lower()
    df.set_index("drugName", inplace = True)
    result = df.loc[medicine]
   
    return result.head(1).to_dict()