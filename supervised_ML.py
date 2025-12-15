#Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")

#Timer
t0 = 0
def eta(t=None):
    global t0
    if t is not None:
        t0 = time.time()
        return
    else:
        t1 = time.time()
        t = t1 - t0
        t0 = t1
        hours, rem = divmod(t, 3600)
        minutes, seconds = divmod(rem, 60)
        return("Ellapsed time {:0>2}:{:0>2}:{:06.3f}".format(int(hours),int(minutes),seconds))


eta(0)

#Read the .csv file
file = pd.read_csv('file.csv', sep=",")
file_t = file.set_index('tenmers').transpose()
print (file_t.shape)
print(file_t['Class'].value_counts())


#Split into features and target class
X = file_t.drop(['Class'], axis=1)
y = file_t.Class

#Models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))

#Split into train and validation sets
train, test = train_test_split(file_t, test_size = 0.2, stratify = file_t.Class, random_state = 42)
#train.to_csv('Train_set.csv', index=True)
#test.to_csv('Test_set.csv', index=True)

print ('Train data dimensions: ',train.shape,'\n',
       'Train data labels: ','\n', train,'\n',
       'Validation data labels: ','\n', valid,'\n',
       'Test data dimensions: ', test.shape,'\n',
       'Test data labels: ','\n', test,'\n')

#Prediction and accuracy calculation
names = []
scores = []
for name, model in models:
    model.fit(train, y_train)
    y_pred = model.predict(test)
    print(y_pred)
    scores.append((accuracy_score(test, y_pred))*100)
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

fig1 = plt.figure(figsize=(5,5))
#Plot accuracies from different algorithms
plt.bar(names,scores, color=(0.4,0.5,0.6))
plt.title('Algorithm Selection')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy (in %)')
for i in range(len(scores)):
    plt.text(x = names[i],y = scores[i], s=round(scores[i],2))
plt.ylim(0,110)
plt.savefig('algorithm_selection.pdf')


#Feature selection
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)*100
print (score)
feature_scores = pd.Series(model.feature_importances_, index=X_train_v.columns).sort_values(ascending=False)
print (feature_scores)
print (len(feature_scores))
feature_scores.to_csv('kmers_importance_scores_021324.csv')

print (eta())



