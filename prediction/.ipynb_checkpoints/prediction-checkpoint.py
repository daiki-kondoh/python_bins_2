import pandas as pd
import numpy as np
import sys
import pprint
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from util.util import lists_to_dict,sort_dict_by_value,extract_dict
import random
random.seed(314)


def kfold_score_RandomForestClassifier(x,y,n_split):
    y=np.reshape(y,-1)
    cv = KFold(n_splits=n_split, random_state=1, shuffle=True)
    prediction_model = RandomForestClassifier(random_state=1)
    scores_list = cross_val_score(prediction_model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    return scores_list,prediction_model

def cul_importance(x,y,model,num):
    y=np.reshape(y,-1)
    labels = list(x.columns)
    model.fit(x,y)
    importances_list = model.feature_importances_
    importances_dict=lists_to_dict(labels,importances_list)
    importances_dict_sorted=sort_dict_by_value(importances_dict,reverse=True)

    result = extract_dict(importances_dict_sorted,num)
    
    return result

def print_confusion_matrix(x,y,model,labels):
    y_test,y_pred=return_predict_true(x,y,model)
    cm=confusion_matrix(y_test, y_pred,labels=labels)
    
    return cm

def compare_pred(x,y,model):
    y_test,y_pred=return_predict_true(x,y,model)
    result=y_pred==y_test

    return result

def return_predict_true(x,y,model):
    y=np.reshape(y,-1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    
    return y_test,y_pred

def gridsearch(x,y,model,scoring):
    gridsearch = GridSearchCV(estimator = model,        # モデル
                          param_grid = param(),  # チューニングするハイパーパラメータ
                          scoring = "accuracy"      # スコアリング
                             )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    gridsearch.fit(x_train, y_train)
    print('Best params: {}'.format(gridsearch.best_params_)) 
    print('Best Score: {}'.format(gridsearch.best_score_))


