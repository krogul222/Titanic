import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *

from collections import Counter

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold, learning_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

#load train and test dataset
train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')

#join train and test datasets before categorical conversion
testID = test["PassengerId"]
trainLength = len(train)
dataset = pd.concat(objs=[train, test], axis = 0).reset_index(drop = True)

#DATA OVERVIEW

#Dataset info 
print("\nDataset Info:")
print(dataset.info())

#Dataset sample - 10 samples
print("\nDataset Sample:")
print(dataset.sample(10))

#Dataset summary
print("\nDataset statistic summary:")
print(dataset.describe())

#MISSING DATA

#AGE
# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
plt.show()

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
plt.show()

#Correlation matrix
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
plt.show()

# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")
plt.show()

# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
indexNaNAge = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in indexNaNAge :
    ageMed = dataset["Age"].median()
    agePred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"])
     & (dataset['Parch'] == dataset.iloc[i]["Parch"])
     & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(agePred) :
        dataset['Age'].iloc[i] = agePred
    else :
        dataset['Age'].iloc[i] = ageMed

#EMBARKED
dataset['Embarked'].value_counts().plot(kind='bar')
plt.xlabel("Embarked")
plt.ylabel("count")
plt.show()
#Filling missing values with most frequent one
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

# convert Embarked into categorical value 0 for S and 1 for Q and 2 for C
dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "Q":1, "C": 2})

#Fare
dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#FEATURE ANALYSIS

#Age
ageBins = (0, 10, 18, 30, 60, 120)
ageGroups = ['Child', 'Teenager','Young Adult', 'Adult', 'Senior']
ageCategories = pd.cut(dataset['Age'], ageBins, labels=ageGroups)
dataset['AgeBinCategories'] = ageCategories
g = sns.factorplot(x = "AgeBinCategories", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()
dataset['AgeBinCode'] = dataset['AgeBinCategories'].map({"Child": 0, "Teenager":1, "Young Adult": 2, 'Adult': 3, 'Senior': 4 })

#Fare"
g = sns.kdeplot(train["Fare"][(train["Survived"] == 0) & (train["Fare"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Fare"][(train["Survived"] == 1) & (train["Fare"].notnull())], color="Blue", shade= True)
g.set_xlabel("Fare")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
plt.show()

fareBins = (-1, 8, 17, 25, 50, 1000)
fareGroups = ['VeryLow', 'Low','Average', 'High', 'VeryHigh']
fareCategories = pd.cut(dataset['Fare'], fareBins, labels=fareGroups)
dataset['FareBinCategories'] = fareCategories
g = sns.factorplot(x = "FareBinCategories", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()
dataset['FareBinCode'] = dataset['FareBinCategories'].map({"VeryLow": 0, "Low":1, "Average": 2, 'High': 3, 'VeryHigh': 4 })

# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})

#FEATURE ENGINEERING

#Name/Title
datasetTitle = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(datasetTitle)
g = sns.countplot(x="Title", data = dataset)
g = plt.setp(g.get_xticklabels(), rotation = 45)
plt.show()
dataset["Title"] = dataset["Title"].replace(['Lady','the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master" : 0, "Miss" : 1, "Ms" : 1, "Mme" : 1, "Mlle" : 1, "Mrs" : 1, "Mr" : 2, "Rare" : 3})

dataset["Title"] = dataset["Title"].astype(int)

g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])

g = sns.factorplot(x = "Title", y = "Survived", data = dataset, kind = "bar")
g = g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Rare"])
g = g.set_ylabels("survival probability")
plt.show()

#Family size
dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.factorplot(x = "FamilySize", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()

#Ticket
ticketGroupCount = Counter(dataset["Ticket"])
ticketGroup = [ticketGroupCount.get(i) for i in dataset['Ticket']]
dataset["TicketSize"] = pd.Series(ticketGroup)
g = sns.factorplot(x = "TicketSize", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("survival probability")
plt.show()

#Groups
dataset["GroupSize"] = dataset[["FamilySize", "TicketSize"]].max(axis=1)
g = sns.factorplot(x = "GroupSize", y = "Survived", data = dataset, kind = "bar")
g = g.set_ylabels("Survival Probability")
plt.show()

#Drop unused data
dataset.drop(labels = ["Cabin"], axis = 1, inplace = True)
dataset.drop(labels = ["FamilySize"], axis = 1, inplace = True)
dataset.drop(labels = ["TicketSize"], axis = 1, inplace = True)
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
dataset.drop(labels = ["Parch"], axis = 1, inplace = True)
dataset.drop(labels = ["SibSp"], axis = 1, inplace = True)
dataset.drop(labels = ["Ticket"], axis = 1, inplace = True)
dataset.drop(labels = ["Age"], axis = 1, inplace = True)
dataset.drop(labels = ["Fare"], axis = 1, inplace = True)
dataset.drop(labels = ["AgeBinCategories"], axis = 1, inplace = True)
dataset.drop(labels = ["FareBinCategories"], axis = 1, inplace = True)

#scaling data
scaler = MinMaxScaler() # StandardScaler() 
dataset[['FareBinCode','AgeBinCode','Pclass', 'GroupSize','Embarked']] = scaler.fit_transform(dataset[['FareBinCode','AgeBinCode','Pclass', 'GroupSize','Embarked']])

#Modeling

#Separate train and data
train = dataset[:trainLength]
test = dataset[trainLength:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"], axis = 1)

# Cross validate model with Kfold cross val
kfold = KFold(n_splits=5, random_state=22)

# Modeling step Test different algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs = 4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans" : cv_means, "CrossValerrors": cv_std, "Algorithm" : ["SVC", "KNeighbors", "RandomForest", "DecisionTree", "ExtraTrees", "AdaBoost", "GradientBoosting"]})

g = sns.barplot("CrossValMeans", "Algorithm", data = cv_res, palette="Set3", orient = "h", **{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

plt.show()
dataset.info()

#Tuning

#RandomForest tuning
RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "max_features": [2, 4, 6, 'auto'],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "n_estimators" :[100, 200 ,400],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,Y_train)
RFC_best = gsRFC.best_estimator_

# Best score
print("Best RF score:\n")
print(gsRFC.best_score_)

# SVC tuning
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose = 1)
gsSVMC.fit(X_train,Y_train)
SVMC_best = gsSVMC.best_estimator_

# Best score
print("Best SVMC score:\n")
print(gsSVMC.best_score_)

# Gradient boosting tunning
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300, 400],
              'learning_rate': [0.1, 0.05, 0.01, 0.001],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.2, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train,Y_train)
GBC_best = gsGBC.best_estimator_

# Best score
print("Best Gradient score:\n")
print(gsGBC.best_score_)

# KNN tunning
KNN = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [1,2,3,4,5,6,7], #default: 5
            'weights': ['uniform', 'distance'], #default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }

gsKNN = GridSearchCV(KNN,param_grid = knn_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsKNN.fit(X_train,Y_train)
KNN_best = gsKNN.best_estimator_

# Best score
print("Best KNN score:\n")
print(gsKNN.best_score_)


#Ensemble modelling

votingC = VotingClassifier(estimators=[('rfc', RFC_best),
('svc', SVMC_best), ('gbc', GBC_best), ('knn', KNN_best)], voting='hard', n_jobs=1)
votingC = votingC.fit(X_train, Y_train)

#Prediction

test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([testID,test_Survived],axis=1)
results.to_csv("Titanic_ensemble_voting.csv",index=False)
