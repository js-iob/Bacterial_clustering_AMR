#Author: K. T. Shreya Parthasarathi
#Script: Classification model tuning and saving random forest model

#Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import time
#import seaborn as sns 
import warnings
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
warnings.filterwarnings("ignore")




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

features = pd.read_csv('selected_kmers.txt', sep = '\t')
selected_kmers = list(features['tenmers'])
print (selected_kmers)
'''
file = pd.read_csv('Train_set.csv', sep=",", usecols=selected_kmers)
print (file.shape)
print(file['Cluster'].value_counts())

#new_train = file[[selected_kmers]]
#new_train['Organism'] = file['Organism']
#new_train['Cluster'] = file['Cluster']
file.to_excel('Train_set_selected_features.xlsx', index = False)


#Split into features and target class
X = file.drop(['Organism','Cluster'], axis=1)
y = file.Cluster


#Split into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)

print ('Train data dimensions: ',X_train.shape,'\n'
       ,'Validation data dimensions: ', X_valid.shape,'\n'
       ,'Train data labels: ','\n', y_train,'\n'
       ,'Validation data labels: ','\n', y_valid,'\n')

rf_param_grid = {'n_estimators':[100,1000], 'max_depth':[2,4,6,8],'criterion':['gini','entropy'], 'min_samples_leaf':[1,2,3,4,5], 'min_samples_split':[2,3,4,5] }
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv = 5, verbose = True)
rf_model = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
predictions = rf_model.predict(X_valid)
accuracy = accuracy_score(y_valid,predictions)*100
print (accuracy)

model = rf_model.fit(X_train, y_train)
joblib.dump(model, 'rf_model.sav')

'''
#Model testing
df = pd.read_csv('Test_set.csv', sep=",", usecols=selected_kmers)
print (df.shape)
y_test = df[['Cluster']]
X_test = df.drop(['Organism','Cluster'], axis=1)
loaded_model = joblib.load('rf_model.sav')
y_predict = loaded_model.predict(X_test)
result = loaded_model.score(X_test,y_test)
print (result)
cm = metrics.confusion_matrix(y_test, y_predict)
target_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5','Cluster 6']
print (classification_report(y_test, y_predict, target_names = target_names))
print ('\n\n')

res = []
for l in [0,1,2,3,4,5,6]:
	pred,recall,_,_=precision_recall_fscore_support(np.array(y_test)==l, np.array(y_predict)==l,pos_label=True,average=None)
	res.append([l,recall[0],recall[1]])
	final_res = (pd.DataFrame(res,columns = ['cluster', 'specificity', 'sensitivity'])
print (final_res)
print (eta())



