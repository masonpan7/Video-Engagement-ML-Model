import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def engagement_model():
    from sklearn.ensemble import RandomForestClassifier

    df_train = pd.read_csv('./assets/train.csv')
    df_test = pd.read_csv('./assets/test.csv')

    most_importance = ['document_entropy', 'freshness', 'easiness',
                    'fraction_stopword_presence', 'silent_period_rate']

    X_train, y_train = df_train[most_importance], df_train.iloc[:,-1].astype(int)
    X_test = df_test[most_importance]

    rfc = RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=4, 
                                min_samples_split=10, n_estimators=200, n_jobs=-1,
                                random_state=0)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict_proba(X_test)

    indexes = df_test['id'].values
    probabilities = y_pred[:,1]

    result = pd.Series(probabilities, index=indexes)

    return result 
    
engagement_model()

#Check if MLP Classifier was a better model for this set
from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    
    
    df = pd.read_csv('assets/train.csv')
    df_test = pd.read_csv('assets/test.csv')
    x_values = df[['title_word_count', 'document_entropy', 'freshness', 'easiness', 'fraction_stopword_presence', 'speaker_speed', 'silent_period_rate']]
    y_values = df['engagement']
    X_test = df_test[['title_word_count', 'document_entropy', 'freshness', 'easiness', 'fraction_stopword_presence', 'speaker_speed', 'silent_period_rate']]
    
    x_values_scaled = scaler.fit_transform(x_values)
    
    #Code to figure out which type of activiation is the most successful
    '''
    for this_activation in ['logistic', 'tanh', 'relu']:
        nnclf = MLPClassifier(hidden_layer_sizes = [50, 50], activation = this_activation, solver = 'lbfgs',
                          alpha = 5, random_state = 0).fit(x_values_scaled, y_values)
    
        train_score = nnclf.score(x_values_scaled, y_values)
    
        print("Train score of the accuracy for {}: {:.2f}".format(this_activation, train_score))
    '''
    
    nnclf = MLPClassifier(hidden_layer_sizes = [50, 50], activation = 'tanh', solver = 'lbfgs',
                          alpha = 5, random_state = 0).fit(x_values_scaled, y_values)
    
    X_test_scaled = scaler.fit_transform(X_test)
    
    probability = nnclf.predict_proba(X_test_scaled)
    stu_ans = pd.Series(probability[:, 1], index = df_test['id']) 
    
    return stu_ans

#Determined the most important features for engagement in a video
from sklearn.feature_selection import SelectKBest, f_classif

X = df_train.iloc[:,1:-1]
y = df_train.iloc[:,-1]

this_k = 8
selector = SelectKBest(f_classif, k='all')
selector.fit(X, y)

# get the score for each feature
scores = selector.scores_

feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
total = feature_scores['Score'].sum()
feature_scores['Score'] = feature_scores['Score']/total
feature_scores.sort_values('Score', ascending=False, inplace=True)

plt.figure(figsize=(6,3))
sns.barplot(x='Score', y='Feature', data=feature_scores)
plt.xlabel('Score')
plt.ylabel('Features')
plt.title('Features importance (Normalized)')
# plt.xticks(rotation=20, ha='right')
plt.show()
