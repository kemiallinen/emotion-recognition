# time for keras

from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier


def create_baseline():
    model = Sequential()
    model.add(Dense(64, input_dim=30, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


data = read_csv('CompleteData_chunks.csv', sep=';').drop(['Speaker', 'Utterance',
                                                          'Chunk', 'PSD Filename',
                                                          'Spectrogram Filename'], axis=1)
sex = 'F'
scaler = StandardScaler()
data_partial = data[data['Sex'] == sex].drop('Sex', axis=1)
y = data_partial['EmoState']
X = scaler.fit_transform(data_partial.drop('EmoState', axis=1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)

seed = 13
scoring = 'accuracy'

estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=8, verbose=0)
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

# cv_results = cross_val_score(estimator, X_train, y_train, cv=kfold)
# print('cv_results: {}\n'
#       'Mean: {} // Std: {}'.format(cv_results, round(cv_results.mean(), 2), round(cv_results.std(), 2)))

