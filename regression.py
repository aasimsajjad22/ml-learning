import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forcast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forcast_out = int(math.ceil(0.1 * len(df)))
#print(forcast_out)

df['label'] = df[forcast_col].shift(-forcast_out)


X = np.array(df.drop(['label', 'Adj. Close'], 1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

#print(len(X), len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = LinearRegression()
clf.fit(X_train, y_train)

# use this for optimization
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# pickle_in = open('linearregression.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
#print(accuracy)

forcast_set = clf.predict(X_lately)

print(forcast_set, accuracy, forcast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#print("last date" + last_date)

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
