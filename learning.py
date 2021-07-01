import numpy as np # linear algebra
import pandas as pd # data processing, CSV file 
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score




from subprocess import check_output

def append_new(new_input):
    from csv import writer

    with open('actions/survey.csv', 'a') as f_object:
      
        writer_object = writer(f_object, lineterminator = '\n')
        writer_object.writerow(new_input)
        print("success")
        f_object.close()

    return

def process_predict():
    train_df = pd.read_csv('actions/survey.csv')
    train_df = train_df.drop(['comments'], axis= 1)
    train_df = train_df.drop(['state'], axis= 1)
    train_df = train_df.drop(['Timestamp'], axis= 1)

    train_df.isnull().sum().max()
    train_df.head(5)


    defaultInt = 0
    defaultString = 'NaN'
    defaultFloat = 0.0

    # Create lists by data type
    intFeatures = ['Age']
    stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                    'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                    'seek_help']
    floatFeatures = []

    # Clean the NaN's
    for feature in train_df:
        if feature in intFeatures:
            train_df[feature] = train_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            train_df[feature] = train_df[feature].fillna(defaultString)
        elif feature in floatFeatures:
            train_df[feature] = train_df[feature].fillna(defaultFloat)
        else:
            print('Error: Feature %s not recognized.' % feature)
   

    #cleaning gender col
    gender = train_df['Gender'].str.lower()
    

    #counting unique elements
    gender = train_df['Gender'].unique()

    #Made gender groups
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
    trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
    female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

    for (row, col) in train_df.iterrows():

        if str.lower(col.Gender) in male_str:
            train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

        if str.lower(col.Gender) in female_str:
            train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

        if str.lower(col.Gender) in trans_str:
            train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)


    stk_list = ['A little about you', 'p']
    train_df = train_df[~train_df['Gender'].isin(stk_list)]



    #complete missing age with mean
    train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

    s = pd.Series(train_df['Age'])
    s[s<18] = train_df['Age'].median()
    train_df['Age'] = s
    s = pd.Series(train_df['Age'])
    s[s>120] = train_df['Age'].median()
    train_df['Age'] = s

    #Ranges of Age
    train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

    #Replace "NaN" string from defaultString
    train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')
  
    train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], 'Don\'t know' )
    #print(train_df['work_interfere'].unique())



    #Encoding data
    labelDict = {}
    for feature in train_df:
        le = preprocessing.LabelEncoder()
        le.fit(train_df[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        train_df[feature] = le.transform(train_df[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] =labelValue
        
    #for key, value in labelDict.items():     
    #    print(key, value)

    #Get rid of 'Country'
    train_df = train_df.drop(['Country'], axis= 1)

    #missing data
    total = train_df.isnull().sum().sort_values(ascending=False)
    percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)
    #print(missing_data)

    # Scaling Age
    scaler = MinMaxScaler()
    train_df['Age'] = scaler.fit_transform(train_df[['Age']])
    train_df.head()

    # define X and y
    feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
    X = train_df[feature_cols]
    y = train_df.treatment

    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    #####################
    def evalClassModel(model, y_test, y_pred_class, plot=False):
        
        #Confusion matrix
        # save confusion matrix and slice into four pieces
        confusion = metrics.confusion_matrix(y_test, y_pred_class)
        #[row, column]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        
  
    def tuningRandomizedSearchCV(model, param_dist):
        rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
        rand.fit(X, y)
        best_scores = []
        for _ in range(20):
            rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10)
            rand.fit(X, y)
            best_scores.append(round(rand.best_score_, 3))
  
    def Knn():
        knn = KNeighborsClassifier(n_neighbors=5)
        
        k_range = list(range(1, 31))
        weight_options = ['uniform', 'distance']
        
        param_dist = dict(n_neighbors=k_range, weights=weight_options)
        tuningRandomizedSearchCV(knn, param_dist)
        
        knn = KNeighborsClassifier(n_neighbors=27, weights='uniform')
        knn.fit(X_train, y_train)
        
        y_pred_class = knn.predict(X_test)
        prediction=knn.predict([[0.3,0,0,2,0,0,0,4]])   
        print(prediction)
        accuracy_score = evalClassModel(knn, y_test, y_pred_class, True)

    def treeClassifier():
        tree = DecisionTreeClassifier()
        featuresSize = feature_cols.__len__()
        param_dist = {"max_depth": [3, None],
                "max_features": randint(1, featuresSize),
                "min_samples_split": randint(2, 9),
                "min_samples_leaf": randint(1, 9),
                "criterion": ["gini", "entropy"]}
        tuningRandomizedSearchCV(tree, param_dist)
        
        tree = DecisionTreeClassifier(max_depth=3, min_samples_split=8, max_features=6, criterion='entropy', min_samples_leaf=7)
        tree.fit(X_train, y_train)
        
        y_pred_class = tree.predict(X_test)
        print(y_pred_class)
        
        listy=X.iloc[-1][feature_cols].values.tolist()
        prediction=tree.predict([listy])
        print(prediction)
        #prediction=tree.predict([[0.3,0,0,2,0,0,0,4]])   
    def boosting():
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        boost = AdaBoostClassifier(base_estimator=clf, n_estimators=500)
        boost.fit(X_train, y_train)
        
]       listy=X.iloc[-1][feature_cols].values.tolist()
        prediction=boost.predict([listy])
        print(prediction)
        return prediction

    def randomForest():
        # Calculating the best parameters
        forest = RandomForestClassifier(n_estimators = 20)

        featuresSize = feature_cols.__len__()
        param_dist = {"max_depth": [3, None],
                "max_features": randint(1, featuresSize),
                "min_samples_split": randint(2, 9),
                "min_samples_leaf": randint(1, 9),
                "criterion": ["gini", "entropy"]}
        tuningRandomizedSearchCV(forest, param_dist)
        
        forest = RandomForestClassifier(max_depth = None, min_samples_leaf=8, min_samples_split=2, n_estimators = 20, random_state = 1)
        my_forest = forest.fit(X_train, y_train)
        
        y_pred_class = my_forest.predict(X_test)
        
        
        accuracy_score = evalClassModel(my_forest, y_test, y_pred_class, True)
    dec=boosting()
    return dec