import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
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
        'feature': X_test.columns,
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

def plot_ROC_curve(model, X_test, y_test):
    '''
    This function calculates the ROC curve and plots it.
    '''
    # calculate the ROC curve
    y_score = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score[:,1])

    # plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()