import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

def plot_confusion_matrix(model, y_test, y_pred):
    '''
    This function calculates the confusion matrix and plots it.
    '''
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_,
                                )
    disp.plot()

def plot_feature_importance(model, X_test, y_test, n_repeats=10, random_state=42):
    '''
    This function calculates the feature importance using permutation importance and plots the top 10 features.
    '''
    # calculate the feature importance using permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state)

    # create a DataFrame to store the feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance['importances_mean'],
        'importance_std': perm_importance['importances_std']
    })

    # sort the features by importance
    feature_importance = feature_importance.sort_values(by='importance_mean', ascending=False)

    # plot the feature importance

    plt.figure(figsize=(10, 5))
    sns.barplot(data=feature_importance.head(10), x='importance_mean', y='feature')
    plt.title('Top 10 Features Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

