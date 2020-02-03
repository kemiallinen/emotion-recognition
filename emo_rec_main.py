import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC as svc
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.linear_model import SGDClassifier as sgdc
from sklearn.naive_bayes import GaussianNB as gnbc
from sklearn.neural_network import MLPClassifier as mlpc


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

models = (('DTC', dtc()), ('SVM', svc(C=10)),
          ('KNN', knc(n_neighbors=10)), ('SGDC', sgdc()),
          ('GNBC', gnbc()), ('MLPC', mlpc(max_iter=1000, learning_rate='adaptive')))
results = []
names = []
seed = 13
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('{}: {} ({})'.format(name, round(cv_results.mean(), 2), round(cv_results.std(), 2)))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
