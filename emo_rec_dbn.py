# https://medium.com/analytics-army/deep-belief-networks-an-introduction-1d52bb867a25
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


data = pd.read_csv('CompleteData_chunks.csv', sep=';').drop(['Speaker', 'Utterance',
                                                             'Chunk', 'PSD Filename',
                                                             'Spectrogram Filename'], axis=1)

sex = 'F'
scaler = StandardScaler()

data_partial = data[data['Sex'] == sex].drop('Sex', axis=1)

y = data_partial['EmoState']
X = scaler.fit_transform(data_partial.drop('EmoState', axis=1))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=71)

rbm = BernoulliRBM(batch_size=10, n_iter=100, verbose=0)
logistic = LogisticRegression(max_iter=10000, tol=0.1)
model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
params = {'rbm__n_components': (64, 80, 100, 128, 200, 256),
          'rbm__learning_rate': (1, 0.1, 0.01, 0.001)}
clf = model_selection.GridSearchCV(model, params, scoring='accuracy')

print('\n. . . : : : F I T T I N G : : : . . .\n')
clf.fit(X_train, y_train)
print('Model performance:\n%s\n' %
      (classification_report(y_test, model.predict(X_test))))
