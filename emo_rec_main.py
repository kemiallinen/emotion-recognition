import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
# TODO: https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin
# TODO: https://scikit-learn.org/stable/modules/classes.html


def plot_corr_matrices(cm1, cm2):
    import matplotlib.pyplot as plt
    fig, (axf, axm) = plt.subplots(1, 2)
    axf.matshow(cm1 ** 2, interpolation='nearest')
    axm.matshow(cm2 ** 2, interpolation='nearest')

    for subf in (axf, axm):
        subf.set_xticks(np.arange(len(cm1)))
        subf.set_yticks(np.arange(len(cm1)))
        subf.set_xticklabels(cm1.columns)
        subf.set_yticklabels(cm1.columns)
        subf.tick_params(axis='both', labelsize=8)
        plt.setp(subf.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")
    plt.show()


data = pd.read_csv('CompleteData_chunks.csv', sep=';').drop(['Speaker', 'Utterance',
                                                             'Chunk', 'PSD Filename',
                                                             'Spectrogram Filename'], axis=1)

sex = 'F'
scaler = StandardScaler()

data_partial = data[data['Sex'] == sex].drop('Sex', axis=1)
# corr_matrix_f, corr_matrix_m = data_f.corr(), data_m.corr()
# plot_corr_matrices(corr_matrix_f, corr_matrix_m)

y = data_partial['EmoState']
X = scaler.fit_transform(data_partial.drop('EmoState', axis=1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)

# SUPPORT VECTOR MACHINES
clf_svm_svc = svm.SVC()
clf_svm_nusvc = svm.NuSVC()
clf_svm_linearsvc = svm.LinearSVC

# LINEAR MODELS
# Stochastic Gradient Descent
# Passive Aggresive Classifier
clf_sgd = SGDClassifier()
clf_pac = PassiveAggressiveClassifier()

# DECISION TREES
clf_tree_dtc = tree.DecisionTreeClassifier()
clf_tree_etc = tree.ExtraTreeClassifier()

# KNN
clf_knn = KNeighborsClassifier()
