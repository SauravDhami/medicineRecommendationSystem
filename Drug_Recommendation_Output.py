#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


# In[2]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')


# In[3]:


medicine = pd.read_csv('test.csv')


# In[4]:


medicine.head()


# In[5]:


df = medicine.iloc[20000:40000]
df.head()


# In[6]:


df.dropna(inplace=True)


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df['review']


# In[10]:


new_list = [i.replace('"','')for i in (df['review'].tolist())]
new_list


# In[11]:


df['review'] = new_list


# In[12]:


df['review'] = df['review'].str.replace('[#,@,&,;,$,0-9]',' ')
df.head()


# In[13]:


df = df[['uniqueID','drugName','condition','review','rating','usefulCount']]
df.head()


# In[14]:


df['review'] = df['review'].apply(lambda x:x.lower())


# In[15]:


df.head()


# In[16]:


lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')


# In[17]:


def text_prep(x: str) -> list:
     corp = str(x).lower() 
     corp = re.sub('[^a-zA-Z]+',' ', corp).strip() 
     tokens = word_tokenize(corp)
     words = [t for t in tokens if t not in stop_words]
     lemmatize = [lemma.lemmatize(w) for w in words]
    
     return lemmatize


# In[18]:


preprocess_tag = [text_prep(i) for i in df['review']]
df["preprocess_txt"] = preprocess_tag


# In[19]:


df['total_len'] = df['preprocess_txt'].map(lambda x: len(x))


# In[20]:


df.head()


# In[21]:


df['drugName'] = df['drugName'].str.lower()


# In[22]:


df.head()


# In[23]:


df['drugName'] = df['drugName'].str.replace('[#,@,&,;,$,/]',' ')

df.head()


# # Sentiment Analysis

# In[24]:


file = open('negative-words.txt', 'r')
neg_words = file.read().split()
file = open('positive-words.txt', 'r')
pos_words = file.read().split()


# In[25]:


num_pos = df['preprocess_txt'].map(lambda x: len([i for i in x if i in pos_words]))
df['pos_count'] = num_pos
num_neg = df['preprocess_txt'].map(lambda x: len([i for i in x if i in neg_words]))
df['neg_count'] = num_neg


# In[26]:


#degree of positivity -> positive_negative_ratio. Range 0 - infinity.  
df['sentiment'] = round(df['pos_count'] / (df['neg_count']+1), 2)


# In[27]:


df.iloc[0:10]


# In[28]:


df['review'][20000]


# # Calculating Weighted Average

# In[29]:


df1 = df[['uniqueID','drugName','condition','review','rating','usefulCount','sentiment']]
df1.head()


# In[30]:


meanRating = df1['rating'].mean()
meanRating


# In[31]:


minUsefulCount = df1['usefulCount'].quantile(0.60)
minUsefulCount


# In[32]:


minSentiment = df1['sentiment'].quantile(0.80)
minSentiment


# In[33]:


df2 = df1.copy().loc[df1['usefulCount'] >= minUsefulCount]  
df2.shape


# In[34]:


df3 = df2.copy().loc[df1['sentiment'] >= minSentiment]
df3.shape


# #  Computing Weighted Avg

# In[35]:


def weighted_rating(X, minUsefulCount=minUsefulCount, meanRating=meanRating):
    usefulCount = X['usefulCount']
    rating = X['rating']
    return (usefulCount/(usefulCount+minUsefulCount) * rating) + (minUsefulCount/(minUsefulCount+usefulCount) * meanRating)


# Calculating weighted average score 

# In[36]:


df3['score'] = df3.apply(weighted_rating, axis = 1)
df3.head()


# Sort dataframe in descending order

# In[37]:


df3 = df3.sort_values('score', ascending = False)


# In[38]:


pd.set_option('display.precision',0)
df3.head(10)


# In[39]:


df3['review'][31456]


# In[40]:


X_test = df3[['uniqueID','rating','usefulCount','sentiment']]
X_test


# Using Model ClassifierRF_Sentiment for Drug Recommendation 

# In[41]:


import joblib
classifier = joblib.load('ClassifierRF Model')


# In[42]:


y_predicted = classifier.predict(X_test)
y_predicted


# In[43]:


y_predicted[0]


# In[44]:


df3['prediction'] = y_predicted
df3.head()


# In[45]:


df3 = df3.sort_values('prediction', ascending = False)


# In[46]:


pd.set_option('display.precision',1)
df3.head(10)


# In[47]:


# condition= df3[df3['condition'] == 'Depression']
# condition.head(5).drugName


# In[48]:


# condition= df3[df3['drugName'] == 'Cymbalta'].condition
# condition.values[0]


# In[49]:


# drugName = df3[df3['condition'] == condition.values[0]]
# drugName.head(5).drugName


# In[50]:


# drugName['drugName'].values[0:6]


# In[55]:


def recommend(medicine):
    medicine =  medicine.lower()
    condition= df3[df3['drugName'] == medicine].condition
    condition.values[0]
    medicineName = df3[df3['condition'] == condition.values[0]]
    medicine_list = list(medicineName['drugName'].values[0:6])
   
    return medicine_list


# In[56]:


recommend('Cymbalta')


# In[ ]:


import pickle
pickle.dump(df3, open('drug.pkl','wb'))


# In[ ]:


# pickle.dump(df3.to_dict(), open('drug_dict.pkl', 'wb'))

#backend

#drug_dict = pickle.load(open('drug_dict', 'rb'))
#drug = pd.DataFrame(drug_dict)


# In[ ]:


# frontend
#  selectbox( drug['drugName'].values)


# In[ ]:


# def recommendsymp(condition):
# # #     medicine =  medicine.tolower()
# #     condition= df3[df3['drugName'] == medicine].condition
# #     condition.values[0]
#     medicineName = df3[df3['condition'] == condition]
#     medicine_list = list(medicineName['drugName'].values[0:6])
   
#     for i in medicine_list:
#         print(i)

