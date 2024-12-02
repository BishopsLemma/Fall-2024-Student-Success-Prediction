
#import data handling libraries
import pandas as pd
import numpy as np
import json
import pickle
import os
from datetime import datetime

#import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#import machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import uniform

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

random_states = [5917, 656, 4125, 2797, 9936]

#################################################################

#data handling functions

def get_splits(data, features, strat_variable):
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

def get_top_features(fimp_df, threshold):
    """Get the top features in fimp_df based on absolute coefficient/importance."""
    # Calculate mean importance and absolute importance
    means = fimp_df.mean()
    abs_means = fimp_df.abs().mean()
    
    # Sort by absolute magnitude but keep original values
    means = means.reindex(abs_means.sort_values(ascending=False).index)
    
    # Filter by threshold
    top_features = means[abs_means > threshold]
    
    return top_features.index.tolist()

def get_mean_accuracies(metrics_dict):
    """Get the mean train and test accuracy for each model in metrics_dict."""
    means_df = pd.DataFrame(columns=['cv_accuracy', 'test_accuracy','accuracy_change'], index=metrics_dict.keys())
    for model, metrics_df in metrics_dict.items():
        means_df.loc[model] = [
            (metrics_df['cv_accuracy'].mean() * 100).round(2),
            (metrics_df['test_accuracy'].mean() * 100).round(2),
            ((metrics_df['accuracy_change'] / metrics_df['cv_accuracy']).mean() * 100).round(2)
        ]
    return means_df

def save_models(models_dict) -> None:
    """
    Save multiple models to file. The parameter "models_dict" is a dictionary with key = model_name and value = list of models. Save each model using the following filename format: f"../Data/Models/{model_name}_{i}.pkl".
    """
    #For each model_name, create a folder with the model_name
    for model_name in models_dict.keys():
        os.makedirs(f"../Data/Models/{model_name}", exist_ok=True)
    for model_name, models in models_dict.items():
        for i, model in enumerate(models):
            filename = f"../Data/Models/{model_name}/{model_name}_{i}.pkl"
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

def load_models(model_names) -> dict:
    """
    Load multiple models from file. The parameter "model_names" is a list of model names. Load each model using the following filename format: f"../Data/Models/{model_name}/{model_name}_{i}.pkl". Return a dictionary with key = model_name and value = list of models.
    """
    models_dict = {}
    for model_name in model_names:
        models = []
        i = 0
        while True:
            filename = f"../Data/Models/{model_name}/{model_name}_{i}.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    model = pickle.load(file)
                    models.append(model)
                i += 1
            else:
                break
        models_dict[model_name] = models
    return models_dict

#################################################################

#Cross-validation with tuning/fitting

def xgb_cv(splits,display=False):
    """ Perform cross-validated hyperparameter tuning for an XGBoost model. Return: 1. a list of tuned models, 2. a dataframe with the cross-validated and test set accuracy, and their difference, 3. a dataframe with the feature importances for each model. """
    #param grid for xgboost randomized search cv
    xgb_param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'lambda': [1, 1.5, 2],
        'alpha': [0, 0.1, 0.2],
        'n_estimators': [150],  # Changed from 'num_boost_round' to 'n_estimators'
        'objective': ['binary:logistic'],
        'eval_metric': ['logloss'],
    }
    models = []
    xgb_df = pd.DataFrame(columns=['cv_accuracy','test_accuracy'])
    fimp_df = pd.DataFrame(columns=splits[0][0].columns)

    #loop over the splits
    for i,(X_train, X_test, y_train, y_test) in enumerate(splits):
        model = xgb.XGBClassifier()
        random_search = RandomizedSearchCV(model,
                                           param_distributions=xgb_param_grid,
                                           n_iter=100,
                                           scoring='accuracy',
                                           verbose=0,
                                           n_jobs=-1,
                                           cv=5,
                                           random_state=random_states[i])
        #fit the model with the best hyperparameters
        random_search.fit(X_train, y_train)
        #add the fitted model to the models list
        models.append(random_search.best_estimator_)

        #add the cross-validated and test set accuracy to the xgb_df dataframe
        y_pred = random_search.predict(X_test)
        xgb_df.loc[i] = [random_search.best_score_,
                         accuracy_score(y_test, y_pred)]
        if display:
            #print the best cross-validated accuracy score and test accuracy score, rounded to 4 decimal places
            print(f'Split {i}')
            print(f'CV accuracy: {random_search.best_score_:0.4f}, Test accuracy: {accuracy_score(y_test, y_pred):0.4f}')

        #add the feature importances (by total_gain) to the fimp_df dataframe. Normalize the feature importances by the sum of the feature importances.
        fimp = random_search.best_estimator_.get_booster().get_score(importance_type='total_gain')
        total_gain = sum(fimp.values())
        for feature in fimp.keys():
            fimp_df.loc[i, feature] = fimp[feature] / total_gain

    #add a column to xgb_df for the drop in accuracy from training to test set
    xgb_df['accuracy_change'] = xgb_df['test_accuracy'] - xgb_df['cv_accuracy']
    fimp_df = fimp_df.astype(float).fillna(0.0, inplace=False)
    #return the models, xgb_df, and fimp_df
    return models, xgb_df, fimp_df

def logreg_cv(splits, display=False):
    """Train LogisticRegression models on multiple splits with RandomizedSearchCV. Return: 1. a list of tuned models, 2. a dataframe with the cross-validated and test set accuracy and their difference, 3. a dataframe with the feature importances for each model. """

    #param grid for logistic regression grid search cv
    logreg_param_grid = {
            'logisticregression__C': [0.001,0.01,0.1,1,10,100],
            'logisticregression__max_iter': [1000],
            'logisticregression__penalty': ['l2'],
        }
    models = []
    metrics_df = pd.DataFrame(columns=['cv_accuracy','test_accuracy'])
    fimp_df = pd.DataFrame(columns=splits[0][0].columns)
    
    #loop over the splits
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression())
        ])
        
        # Random search
        search = GridSearchCV(pipe, 
            param_grid=logreg_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        # Fit the model with the best hyperparameters
        search.fit(X_train, y_train)
        # Store model
        models.append(search.best_estimator_)
        # Calculate metrics
        y_pred = search.predict(X_test)
        metrics_df.loc[i] = [
            search.best_score_,
            accuracy_score(y_test, y_pred)]

        if display:
            #print the best cross-validated accuracy score and test accuracy score, rounded to 4 decimal places
            print(f'Split {i} best CV accuracy: {search.best_score_:0.4f}, Test accuracy: {accuracy_score(y_test, y_pred):0.4f}')
        
        # Feature importances (coefficients)
        coef = search.best_estimator_['logisticregression'].coef_[0]

        # Add the coefficients to the fimp_df dataframe. 
        fimp_df.loc[i] = coef 
        
    #add a column to metrics_df for the drop in accuracy from training to test set
    metrics_df['accuracy_change'] = metrics_df['test_accuracy'] - metrics_df['cv_accuracy']
    fimp_df = fimp_df.astype(float).fillna(0.0, inplace=False)
    #return the models, metrics_df, and fimp_df
    return models, metrics_df, fimp_df

def svc_rbf_cv(splits, display=False):
    """Train SVC-RBF models on multiple splits with cross-validation. Return: 1. a list of models, 2. a dataframe with the cross-validated and test set accuracy, and their difference. """
    
    models = []
    metrics_df = pd.DataFrame(columns=['cv_accuracy','test_accuracy'])
    
    #loop over the splits
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Create pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', probability=True))
        ])
        
        # Fit the model and get cross-validated accuracy
        pipe.fit(X_train, y_train)
        cv_accuracy = np.mean(cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1,scoring='accuracy'))
        
        # Store model
        models.append(pipe)
        
        # Calculate metrics
        y_pred = pipe.predict(X_test)
        metrics_df.loc[i] = [
            cv_accuracy,
            accuracy_score(y_test, y_pred)]

        if display:
            #print the best cross-validated accuracy score and test accuracy score, rounded to 4 decimal places
            print(f'Split {i} best CV accuracy: {cv_accuracy:0.4f}, Test accuracy: {accuracy_score(y_test, y_pred):0.4f}')
        
    #add a column to metrics_df for the drop in accuracy from training to test set
    metrics_df['accuracy_change'] = metrics_df['test_accuracy'] - metrics_df['cv_accuracy']
    return models, metrics_df

def stacked_classifier(splits, xgb_models, svc_models, display=False):
    """Stack classifiers using logistic regression on probability outputs of SVC with RBF and XGBoost models. Return: 1. a list of fitted meta classifiers, 2. a dataframe with the train and test set accuracy, and their difference. """
    
    metrics_df = pd.DataFrame(columns=['cv_accuracy', 'test_accuracy'])
    meta_classifiers = []
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        #Get base models
        xgb_model = xgb_models[i]
        svc_model = svc_models[i]
        
        # Get probabilities for training set
        proba_train = np.column_stack([
            xgb_model.predict_proba(X_train)[:, 1],
            svc_model.predict_proba(X_train)[:, 1]
        ])
        
        # Get probabilities for test set
        proba_test = np.column_stack([
            xgb_model.predict_proba(X_test)[:, 1],
            svc_model.predict_proba(X_test)[:, 1]
        ])
        
        # Fit meta classifier with StandardScaler
        meta_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression())
        ])
        meta_clf.fit(proba_train, y_train)
        meta_classifiers.append(meta_clf)
        
        # Get predictions
        y_pred_train = meta_clf.predict(proba_train)
        y_pred_test = meta_clf.predict(proba_test)
        
        # Calculate metrics
        metrics_df.loc[i] = [
            accuracy_score(y_train, y_pred_train),
            accuracy_score(y_test, y_pred_test)
        ]
        
        if display:
            #print the best cross-validated accuracy score and test accuracy score, rounded to 4 decimal places
            print(f'Split {i}:')
            print(f'Train accuracy: {accuracy_score(y_train, y_pred_train):0.4f}, Test accuracy: {accuracy_score(y_test, y_pred_test):0.4f}')

    #add a column to metrics_df for the drop in accuracy from training to test set
    metrics_df['accuracy_change'] = metrics_df['test_accuracy'] - metrics_df['cv_accuracy']
    return meta_classifiers, metrics_df

#################################################################

# Functions for plotting feature importances

def xgb_plot_fimp(fimp_df, save=False):
    """ Plot the feature importances (total_gain) for each XGBoost model. """
    #each row of fimp_df contains the normalized feature importances for the corresponding model. For each feature, we take the mean of the normalized feature importances across all models.
    fimp_df = fimp_df.mean().sort_values(ascending=False)
    fimp_df = fimp_df.apply(lambda x: np.round(100*x, 2))

    #plot the feature importances using a seaborn barplot
    plt.figure(figsize=(10, 20))
    sns.barplot(x=fimp_df.values, y=fimp_df.index)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importances for XGBoost Models')

    # Add value labels to the bars
    for index, value in enumerate(fimp_df.values):
        plt.text(value, index, f'{value}%', va='center')
    plt.tight_layout()
    if save:
        plt.savefig('../Data/Feature Importances/Feature_importances_xgb.png')
    plt.show()

def logreg_plot_fimp(fimp_df, save=False):
    """Plot the feature importances (coefficients) for each LogisticRegression model."""
    # Calculate means
    abs_means = fimp_df.abs().mean()
    means = fimp_df.mean()
    
    # Sort by absolute magnitude
    means = means.reindex(abs_means.sort_values(ascending=False).index)
    abs_means = abs_means.reindex(abs_means.sort_values(ascending=False).index)

    # Plot, coloring the bars based on sign
    plt.figure(figsize=(10, 20))
    colors = ['indianred' if x < 0 else 'steelblue' for x in means.values]
    sns.barplot(x=abs_means.values, y=abs_means.index, palette=colors, hue=abs_means.index, legend=False)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.2)
    
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Feature Coefficients for Logistic Regression Models')
    
    # Add coefficient values with adjusted positions
    for index, value in enumerate(means.values):
        x_pos = abs_means.values[index] + 0.01  # Slightly outside bar
        ha = 'left'
        plt.text(x_pos, index, f'{value:.3f}', va='center', ha=ha)
    plt.tight_layout()
    if save:
        plt.savefig('../Data/Feature Importances/Feature_importances_logreg.png')
    plt.show()

#################################################################

# Functions for calculating Bayes error rate

def get_bayes_error(data, features, save=False):
    """ Calculate the Bayes error rate for a list of variables (features) in a dataframe (data). """
    grouped = data.groupby(features)
    data_grouped = pd.DataFrame({
        'COUNT(X)' : grouped.size(),
        'Pr(X)' : grouped.size() / len(data),
        'Pr(Y=1|X)': grouped['Y'].mean()}).reset_index()

    #add a column named "ERROR(Y|X)" which contains the minimum of Pr(Y|X) and 1-Pr(Y|X)
    data_grouped['ERROR(Y|X)'] = np.minimum(data_grouped['Pr(Y=1|X)'], 1 - data_grouped['Pr(Y=1|X)'])

    #add a column named "ERROR(Y|X) * Pr(X)" which is the product of "ERROR(Y|X)" and "Pr(X)"
    data_grouped['ERROR(Y|X) * Pr(X)'] = data_grouped['ERROR(Y|X)'] * data_grouped['Pr(X)']

    #compute the bayes error rate. This is the expected value of ERROR(Y|X) over the distribution of X
    bayes_error_rate = np.dot(data_grouped['ERROR(Y|X)'], data_grouped['Pr(X)'])

    #make a separate dataframe df_singletons which contains only the rows where 'COUNT(X)' is 1
    data_singletons = data_grouped[data_grouped['COUNT(X)'] == 1]

    #drop the rows in df_singletons from data_grouped
    data_grouped = data_grouped[data_grouped['COUNT(X)'] != 1]

    #sort the data_grouped dataframe by Pr(Y|X) in descending order by 'ERROR(Y|X) * Pr(X)', and in case of a tie, by 'COUNT(X)' in descending order
    data_grouped = data_grouped.sort_values(by=['ERROR(Y|X) * Pr(X)', 'COUNT(X)'], ascending=[False, False]).reset_index(drop=True)

    #add a column for the cumulative sum of 'ERROR(Y|X) * Pr(X)' along the column
    data_grouped['CUMULATIVE_ERROR'] = data_grouped['ERROR(Y|X) * Pr(X)'].cumsum()

    if save:
        data_grouped.to_csv('../Data/Datasets/dataset_bayes_grouped.csv')
        data_singletons.to_csv('../Data/Datasets/dataset_bayes_singletons.csv')

    return data_grouped, data_singletons