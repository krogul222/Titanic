# <p align = "center">Titanic Project: Machine Learning from Disaster</p>
## 1. Problem definition
When Titanic hit an iceberg a lot of passengers and crew died. In this challenge I need to predict what sorts of people were likely to survive. Data on which I will do my prediction was included into task.

## 2. Preparation of Data
### 2.1 Data overview
To take first overview of the data let's explore whole dataset (train and test data joined) using:
* info() function
<p align ="center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Info.png?raw=true"></p>
<p align ="center"> <b>Fig 2.1.1</b> Dataset info. </p>

* sample() function
<p align ="center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/SampleFunctionpart1.png?raw=true"></p>
<p align ="center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/SampleFunctionpart2.png?raw=true"></p>
<p align ="center"> <b>Fig 2.1.2</b> Ten samples of dataset. </p>

* description() function
<p align ="center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/DescriptionFunctionpart1.png?raw=true"></p>
<p align ="center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/DescriptionFunctionpart2.png?raw=true"></p>
<p align ="center"><b>Fig 2.1.3</b> Dataset description.</p>

There are 12 features which can be used for data analysis:
- **Survived** is a variable which need to be predicted
- **PassengerID** is assumed to be a random identifier and therefore will be exclded from analysis
- **Pclass** variable represents passenger status: 1 - upper class, 2 - middle class, 3 - lower class
- **Ticket** variable is a nominal datatype that can be used in feature engineering to check which passengers were travelling together (same ticket number)
- **Cabin** vaiable is a nominal dataype  that can be used in feature engineering to approximate passenger's position on the ship when Titanic hit an iceberg
- **Name** vaiable is a nominal dataype  that can be used in feature engineering to obtain passenger title
- **Sex** and **Embarked** variables are a nominal datatypes
- **Age** and **Fare** variables are continuous quantitative datatypes
- **SibSp** represents number of related siblings/spouse on board
- **Parch** represents number of related parents/children on board. Togather with SibSp can be used for feature engineering to create a family size variable

### 2.2 Outliers
Since outliers can have a dramatic effect on the prediction I decided to inspect if all numerical data look reasonable. From all numerical variables I analysed Fare, SibSP, Parch and Age (Pclass is not really  numerical variable but categorised, PassengerID is excluded and Survived is categorical variable which I need to predict).

I based my judgement on statisic summary of the dataset (Fig 2.1.3):
- **Age** values are between 0.17 and 80 which is reasonable
- **Parch** are between 0 and 9  which is reasonable.  Nine  children/parents may look high at first but it wasn't unusual at that time to have many children.
- **SibSP** values are between 0 and 8  which is also reasonable.  
- **Fare** values are between 0 and 512.3292 with mean 33.295479. Maximum value may look high at fist but there were high difference in passenger status. Some of them was really rich and had high quality rooms which could cost significantly higher comparing to standard tickets.

Since all numerical features have reasonable values I decided not to exclude outliers from original data. 

### 2.3 Missing values
There ae missing values in Age, Embarked, Fare and Cabin fields. Before modelling records with missing values should be deleted, fixed or feature should be excluded from analysis. Let analyse each featue one by one.

#### 2.3.1 Age
First of all I am going to plot age distribution for people who survived the tragedy and who died.

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSurvived.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.1.1</b> Age distribution vs Survived.</p>
<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSurvived_two_on_one.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.1.2</b> Age distribution vs Survived. Two distributions in one plot.</p>

There is a clear difference between these two distributions. For “survival” distribution there is a peak for young passengers and group of old people (over 60) seems to be smaller comparing to “non survival” distribution. It looks like age is a valuable feature for survival prediction. Because of that it shouldn’t be excluded from analysis even there is significant amount of missing data in this field. 

To impute the missing age values I used other features to calculate reasonable guess. I based on Parch, SibSp, Pclass and Sex .

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Age correlation matrix.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.1.3</b> Correlation matrix fo Age, SibSp, Parch and Pclass features.</p>

According to correlation matrix age is not correlated with sex but it is negatively correlated with Parch, SibSp and Pclass, positively with Fare. To visualize it more I created aproperiate plots.

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSex.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.1.4</b> Age vs Sex box plots.</p>

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsPclass.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.1.5</b> Age vs Pclass (male and female) box plots.</p>

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsParch.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.1.6</b> Age vs Parch box plots.</p>

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgevsSibSp.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.1.7</b> Age vs SibSp box plots.</p>

I decided to use Parch, SibSp and Pclass features to predict missing age values.

```python
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
```

#### 2.3.2 Cabin
Cabin field is a source of valuable information – where passenger probably was at the time when Titanic hit an iceberg. However, it has a lot of missing values (over 1000) which can introduce a lot of noise in prediction step. Eventually I decided to exclude this feature from future analysis.

#### 2.3.3 Embarked
There are only 2 missing values so I decided to fill them with most frequent value which is “S”.

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Embarked count plot.png?raw=true"></p>
<p align = "center"><b>Fig 2.3.3.1</b> Countplot of Embarked feature.</p>

#### 2.3.4 Fare
There is only 1 missing values so I decided to fill it with median.

## 3. Feature Analysis
### 3.1 Age
Many machine learning algorithms are known to produce better model by discretizing continous features. Because of that fact I decided to arrange Age feature into 5 categories. I based on data from Fig. 2.3.1.1 and chose following categories:
* Child – age between 0 and 10
* Teenager - age between 10 and 18
* Young Adult - age between 18 and 30
* Adult - age between 30 and 60
* Senior – age between 60 and 120

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/AgeBins.png?raw=true"></p>
<p align = "center"><b>Fig 3.1.1</b> Age groups vs survival probability.</p>

It looks like children have more chances to survive comparing to other groups and oldest peapole have lowest less chances. It is consistent with common sense. 

### 3.2 Fare
I decided to discretize Fare feature because of the same reasons I discretized Age feature. I ploted fare distribution for people who survived the tragedy and who died (Fig 3.2.1).

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/FarevsSurvived_two_on_one.png?raw=true"></p>
<p align = "center"><b>Fig 3.2.1</b> Fare distribution vs Survived. Two distributions in one plot.</p>

There is a clear difference between these two distributions. People who paid small amount for ticket (less than 17) had significantly less chance to survive (red peak). It is consistent with common sense. Low ticket fare is probably connected with low status which can be an obstacle during "selection" who can go on life boat and who can't. I decided to arrange Fare feature into 5 categories:

* VeryLow - fare between 0 and 8 (first half of red peak)
* Low - fare between 8 and 17 (second half of red peak)
* Average - fare between 17 and 25
* High - fare between 25 and 50
* VeryHigh - fare between 50 and 1000

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/FareBins.png?raw=true"></p>
<p align = "center"><b>Fig 3.2.2</b> Fare groups vs survival probability.</p>

Fig 3.2.2 confirms my previous conclusions. People with lower ticket fare had less chance to survive. However, there is not much difference between average and high fare but significant difference between very high fare comparing to other groups. It looks like rich people had more chances to survive, probably because of their status.

### 3.3 Other Features

I decided to convert Sex feature into categorical with value 0 for male and 1 for female. Similarly I did with Embarked feature with value 0 for S and 1 for Q and 2 for C. I left Pclass feture unchanged. Rest of the features (SibSP, Parch, Name, Ticket) will be used in feature engineering section.

## 4. Feature Engineering
### 4.1 Title
Name feature despite first and second name includes passenger title. It is interesting to separate title as a feature to see what is survival probability for passengers with different titles.
There are 17 different titles in the dataset but most of them are very rare. They can be grouped into 4 categories: 
- Mr
- Master
- Mrs/ Ms/Mme/ Miss
- Rare (Don, Sir, …)

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/Titles.png?raw=true"></p>
<p align = "center"><b>Fig 4.1.1</b>  Countplot of Title feature.</p>

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/TitlesvsSurvived.png?raw=true"></p>
<p align = "center"><b>Fig 4.1.2</b> Title vs survival probability.</p>

It looks like people with rare title had more chance to survive comparing to male with basic title (Mr.).

### 4.2 Group size
It is possible that size of a family had impact on survival probability of passengers. It is not easy to evacuate big family. It is easy to construct family size feature basing on SibSp and Parch features. “Family” value is sum of SibSp, Parch and 1. 

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/FamilySizevsSurvived.png?raw=true"></p>
<p align = "center"><b>Fig 4.2.1</b> FamilySize vs survival probability.</p>

It looks like single people and large families had significantly lower chances to survive comparing to other groups. However, people do not travel only with close family (SibSp, Parch). Somtimes they travel with other relatives or just friends. How it can be detected? People who travelled togather on Titanic often had the same ticket number. In this case we can define group of people basing on ticket number.

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/TicketSizevsSurvived.png?raw=true"></p>
<p align = "center"><b>Fig 4.2.2</b> TicketSize vs survival probability.</p>

As you can see there are some differences between charts on Fig 4.2.2 and 4.2.1 . I decided to create GroupSize feature basing on TicketSize and FamilySize which is maximum of these to features for each observation.

<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/GroupSizevsSurvived.png?raw=true"></p>
<p align = "center"><b>Fig 4.2.3</b> GroupSize vs survival probability.</p>

It looks like medium size groups (3,4) had significantly more chances than other groups. On the other hand large gorups and singles had less chances.

## 5. Modeling

### 5.1 Data preparation

```python
#Separate train and data
train = dataset[:trainLength]
test = dataset[trainLength:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"], axis = 1)
```

### 5.2 Models cross validation

I compared 7 commonly used classifiers and evaluated the mean accuracy of each of them by a kfold cross validation procedure:

* SVC
* KNN
* RandomForest
* Decision Tree
* Extra Trees
* AdaBoost
* Gradient Boosting

```python
# Cross validate model with Kfold cross val
kfold = KFold(n_splits=5, random_state=22)
```

```python
# Cross validate model with Kfold cross val
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
```
<p align = "center"><img src="https://github.com/krogul222/Data-Science-Projects/blob/master/Python/Titanic/img/ModelsPerformance.png?raw=true"></p>
<p align = "center"><b>Fig 5.2.1</b> Mean models accuracy.</p>

I decided to choose the SVC, RandomForest , KNeighbors and the GradientBoosting classifiers for the ensemble modeling.

### 5.3 Hyperparameter tunning of chosen models

I performed a grid search optimization for classifiers chosen in previous section:

* Random Forest
    
```python
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
```

```python
Best RF score:
0.8327721661054994
```

* SVC

```python
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
```

```python
Best SVMC score:
0.8338945005611672
```

* Gradient boosting

```python
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
```

```python
Best Gradient score:
0.8237934904601572
```

* KNN

```python
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
```

```python
Best KNN score:
0.8215488215488216
```

### 5.4 Ensemble modeling

I choosed a voting classifier to combine the predictions coming from the 4 classifiers. I decided to pass the argument "hard" to the voting parameter so it will by majority rule voting.

```python
#Ensemble modelling

votingC = VotingClassifier(estimators=[('rfc', RFC_best),
('svc', SVMC_best), ('gbc', GBC_best), ('knn', KNN_best)], voting='hard', n_jobs=1)
votingC = votingC.fit(X_train, Y_train)
```

### 5.5 Prediction

```python
#Prediction

test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([testID,test_Survived],axis=1)
results.to_csv("Titanic_ensemble_voting.csv",index=False)
```

After submitting results on Kaggle competition I obtained 0.80382 score which classified me on 1,047 place out of 10,365.