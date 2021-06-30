import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
import re
ps=PorterStemmer()

s1=input("enter comment")
# -------------------------------
corpus3=[]
a=set(stopwords.words('english'))

review=re.sub('[^a-zA-Z]',' ',s1)
review=review.lower()
corpus3.append(review)

corpus4=[]
for ele in corpus3:
  l=list(ele.split())
  review=[ps.stem(word) for word in l if not word in a]
  corpus4.append(' '.join(review))

# ================================

loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

    # load the model
loaded_model = pickle.load(open('classification.model', 'rb'))

    # make a prediction
# print(loaded_model.predict(loaded_vectorizer.transform(['bitch please'])))

res=loaded_model.predict(loaded_vectorizer.transform(corpus4))
print(res)
if 1 in res[0]:
    print("this Text Contains toxic information")
else:
    print("good text")
s=[]
if 1 in res[0]:
    if res[0][0]==1:
        s.append("Toxic")
    if res[0][1]==1:
        s.append("Severe_Toxic")
    if res[0][2]==1:
        s.append("onscene")
    if res[0][3]==1:
        s.append("Threat")
    if res[0][4]==1:
        s.append("Insult")
    if res[0][5]==1:
        s.append("Identity Hate")
    print("Text  contains",",".join(s),"information")

else:
    print("good text")





