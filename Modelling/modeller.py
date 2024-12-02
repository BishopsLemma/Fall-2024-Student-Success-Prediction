
#import data handling libraries
import pandas as pd
import numpy as np
import json

#import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#import machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import uniform, loguniform

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb



def bayes_error_rate(data, features):
    """ Calculate the Bayes error rate for a list of variables (features) in a dataframe (data). """
    grouped = data.groupby(features)
    data_grouped = pd.DataFrame({
        'COUNT(X)' : grouped.size(),
        'Pr(X)' : grouped.size() / len(data),
        'Pr(Y|X)': grouped['Y'].mean()}).reset_index()

    #add a column named "ERROR(Y|X)" which contains the minimum of Pr(Y|X) and 1-Pr(Y|X)
    data_grouped['ERROR(Y|X)'] = np.minimum(data_grouped['Pr(Y|X)'], 1 - data_grouped['Pr(Y|X)'])

    #compute the bayes error rate. This is the expected value of ERROR(Y|X) over the distribution of X
    bayes_error_rate = np.dot(data_grouped['ERROR(Y|X)'], data_grouped['Pr(X)'])
    
    return bayes_error_rate, data_grouped

def get_splits(data,
               features,
               random_states,
               strat_variable):
    """ Get train-test splits for a dataframe (data) using a list of random states (random_states) and a stratification variable (strat_variable). """
    splits = []
    for random_state in random_states:
        X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                            data['Y'],
                                                            test_size=0.2,
                                                            random_state=random_state,
                                                            stratify=data[strat_variable])
        splits.append((X_train, X_test, y_train, y_test))
    return splits

def xgb_cv(splits,
           random_states,
           param_grid):
    """ Perform cross-validated hyperparameter tuning for an XGBoost model. Return: 1. a list of tuned models, 2. a dataframe with the cross-validated and test set accuracy, precision, recall, and F1 score for each model, 3. a dataframe with the feature importances for each model. """

    models = []
    xgb_df = pd.DataFrame(columns=['cv_accuracy','test_accuracy', 'test_precision', 'test_recall', 'test_f1'])
    fimp_df = pd.DataFrame(columns=splits[0][0].columns)
    for i,(X_train, X_test, y_train, y_test) in enumerate(splits):
        model = xgb.XGBClassifier()
        random_search = RandomizedSearchCV(model,
                                           param_distributions=param_grid,
                                           n_iter=100,
                                           scoring='accuracy',
                                           verbose=0,
                                           n_jobs=-1,
                                           cv=5,
                                           random_state=random_states[i])
        random_search.fit(X_train, y_train)

        #add the cross-validated and test set accuracy, precision, recall, and F1 score to the xgb_df dataframe
        y_pred = random_search.predict(X_test)
        xgb_df.loc[i] = [random_search.best_score_,
                         accuracy_score(y_test, y_pred),
                         precision_score(y_test, y_pred),
                         recall_score(y_test, y_pred),
                         f1_score(y_test, y_pred)]
        #print the best cross-validated accuracy score and test accuracy score, rounded to 4 decimal places
        print(f'Split {i} best CV accuracy: {random_search.best_score_:0.4f}, Test accuracy: {accuracy_score(y_test, y_pred):0.4f}')
        #add the feature importances (by total_gain) to the fimp_df dataframe. Normalize the feature importances by the sum of the feature importances.
        fimp = random_search.best_estimator_.get_booster().get_score(importance_type='total_gain')
        total_gain = sum(fimp.values())
        for feature in fimp.keys():
            fimp_df.loc[i, feature] = fimp[feature] / total_gain

        #add the model with the best hyperparameters to the list of tuned models (not the fitted model)
        models.append(xgb.XGBClassifier(**random_search.best_params_))

    #return the models, xgb_df, and fimp_df
    return models, xgb_df, fimp_df


def logreg_cv(splits, random_states, param_grid):
    """Train LogisticRegression models on multiple splits with RandomizedSearchCV. Return: 1. a list of tuned models, 2. a dataframe with the cross-validated and test set accuracy, precision, recall, and F1 score for each model, 3. a dataframe with the feature importances for each model. """
    
    models = []
    metrics_df = pd.DataFrame(columns=['cv_accuracy','test_accuracy', 'test_precision', 'test_recall', 'test_f1'])
    fimp_df = pd.DataFrame(columns=splits[0][0].columns)
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression())
        ])
        
        # Random search
        search = RandomizedSearchCV(
            pipe, param_grid,
            n_iter=100,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_states[i]
        )
        search.fit(X_train, y_train)

        # Calculate metrics
        y_pred = search.predict(X_test)
        metrics_df.loc[i] = [
            search.best_score_,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        ]

        #print the best cross-validated accuracy score and test accuracy score, rounded to 4 decimal places
        print(f'Split {i} best CV accuracy: {search.best_score_:0.4f}, Test accuracy: {accuracy_score(y_test, y_pred):0.4f}')
        
        # Feature importances (coefficients)
        coef = np.abs(search.best_estimator_['logisticregression'].coef_[0])

        # Add the coefficients to the fimp_df dataframe. 
        fimp_df.loc[i] = coef 
        
        # Extract best parameters for LogisticRegression only
        best_params = {key.replace('logisticregression__', ''): value 
                    for key, value in search.best_params_.items() 
                    if key.startswith('logisticregression__')}

        # Create new unfitted pipeline with best params
        unfitted_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression(**best_params))
        ])

        # Add to models list
        models.append(unfitted_pipe)
        
    return models, metrics_df, fimp_df

def svc_rbf_cv(splits, random_states):
    """Train SVC-RBF models on multiple splits with cross-validation. Return: 1. a list of models, 2. a dataframe with the cross-validated and test set accuracy, precision, recall, and F1 score for each model. """
    
    models = []
    metrics_df = pd.DataFrame(columns=['cv_accuracy','test_accuracy', 'test_precision', 'test_recall', 'test_f1'])
    fimp_df = pd.DataFrame(columns=splits[0][0].columns)
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', probability=True))
        ])
        
        # Cross-validation
        pipe.fit(X_train, y_train)
        cv_accuracy = np.mean(cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1,scoring='accuracy'))
        
        # Store model
        models.append(pipe)
        
        # Calculate metrics
        y_pred = pipe.predict(X_test)
        metrics_df.loc[i] = [
            cv_accuracy,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        ]

        #print the best cross-validated accuracy score and test accuracy score, rounded to 4 decimal places
        print(f'Split {i} best CV accuracy: {cv_accuracy:0.4f}, Test accuracy: {accuracy_score(y_test, y_pred):0.4f}')
        
    return models, metrics_df


def xgb_plot_fimp(fimp_df):
    """ Plot the feature importances (total_gain) for each XGBoost model. """
    fimp_df = fimp_df.fillna(0)

    #each row of fimp_df contains the normalized feature importances for the corresponding model. For each feature, we take the mean of the normalized feature importances across all models.
    fimp_df = fimp_df.mean().sort_values(ascending=False)

    #plot the feature importances using a seaborn barplot
    plt.figure(figsize=(10, 20))
    sns.barplot(x=fimp_df.values, y=fimp_df.index)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importances for XGBoost Models')
    plt.show()

def logreg_plot_fimp(fimp_df):
    """ Plot the feature importances (coefficients) for each LogisticRegression model. """
    fimp_df = fimp_df.fillna(0)

    #each row of fimp_df contains the normalized feature importances for the corresponding model. For each feature, we take the mean of the normalized feature importances across all models.
    fimp_df = fimp_df.mean().sort_values(ascending=False)

    #plot the feature importances using a seaborn barplot
    plt.figure(figsize=(10, 20))
    sns.barplot(x=fimp_df.values, y=fimp_df.index)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importances for Logistic Regression Models')
    plt.show()

def get_top_features(fimp_df, n):
    """ Get the top n features in fimp_df. """
    #calculate the mean feature importance for each feature
    fimp_df = fimp_df.mean()
    #sort the features by mean feature importance
    fimp_df = fimp_df.sort_values(ascending=False)
    #get the top n features
    return fimp_df.index[:n].tolist()

def customized_binary_classifier(splits, logreg_models, xgb_models, 
                               xgb_features, logreg_features, svc_features):
    svm_models = []
    results = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        logreg_fitted = []
        xgb_fitted = []
        svc_fitted = []

        # Step 2: Fit SVM model with its features
        X_train_svc = X_train[svc_features]
        X_test_svc = X_test[svc_features]
        
        svm_model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True))
        ])
        svm_model.fit(X_train_svc, y_train)
        svm_predictions_train = svm_model.predict_proba(X_train_svc)[:, 1]
        svm_predictions_test = svm_model.predict_proba(X_test_svc)[:, 1]
        svc_fitted.append(svm_model)

        # Calculate SVM accuracy
        svm_train_accuracy = accuracy_score(y_train, svm_model.predict(X_train_svc))
        svm_test_accuracy = accuracy_score(y_test, svm_model.predict(X_test_svc))

        # Step 3: Fit pre-tuned Logistic Regression model using selected features
        logistic_model = logreg_models[i]
        X_train_log = X_train[logreg_features].copy()
        X_train_log['svm_pred'] = svm_predictions_train
        X_test_log = X_test[logreg_features].copy()
        X_test_log['svm_pred'] = svm_predictions_test
        logistic_model.fit(X_train_log, y_train)
        logistic_predictions_train = logistic_model.predict_proba(X_train_log)[:, 1]
        logistic_predictions_test = logistic_model.predict_proba(X_test_log)[:, 1]
        logreg_fitted.append(logistic_model)

        # Calculate Logistic Regression accuracy
        logistic_train_accuracy = accuracy_score(y_train, logistic_model.predict(X_train_log))
        logistic_test_accuracy = accuracy_score(y_test, logistic_model.predict(X_test_log))

        # Use pre-tuned XGBoost model with its features
        xgb_model = xgb_models[i]
        X_train_xgb = X_train[xgb_features].copy()
        X_train_xgb['logistic_pred'] = logistic_predictions_train
        X_train_xgb['svm_pred'] = svm_predictions_train
        X_test_xgb = X_test[xgb_features].copy()
        X_test_xgb['logistic_pred'] = logistic_predictions_test
        X_test_xgb['svm_pred'] = svm_predictions_test
        
        xgb_model.fit(X_train_xgb, y_train)
        xgb_predictions_train = xgb_model.predict(X_train_xgb)
        xgb_predictions_test = xgb_model.predict(X_test_xgb)
        xgb_fitted.append(xgb_model)

        results.append({
            'logistic_train_accuracy': logistic_train_accuracy,
            'logistic_test_accuracy': logistic_test_accuracy,
            'svm_train_accuracy': svm_train_accuracy, 
            'svm_test_accuracy': svm_test_accuracy,
            'xgb_train_accuracy': accuracy_score(y_train, xgb_predictions_train),
            'xgb_test_accuracy': accuracy_score(y_test, xgb_predictions_test)
        })
        # Print the final train and test accuracies of the xgboost, rounded to 4 decimal places
        print(f'Split {i+1} accuracies for custom model:')
        print(f'Train: {round(results[-1]["xgb_train_accuracy"], 4)}   Test: {round(results[-1]["xgb_test_accuracy"], 4)}')

    return {'results': pd.DataFrame(results),
            'logreg_models': logreg_fitted,
            'xgb_models': xgb_fitted,
            'svm_models': svc_fitted}