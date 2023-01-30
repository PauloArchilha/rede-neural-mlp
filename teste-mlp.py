import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv('credit_data.csv')
df = df.dropna()
# print(df.head())
# print(df.shape)

X_credit = df[['income',  'age', 'loan']]
Y_credit = df[['default']]

ss = StandardScaler()
X_credit = ss.fit_transform(X_credit)
# print(X_credit)

X_credit_train , X_credit_test, Y_credit_train , Y_credit_test = train_test_split(X_credit, Y_credit, test_size= 0.25, random_state= 0)

# print(X_credit_train.shape)
# print(Y_credit_train.shape)
# print(X_credit_test.shape, Y_credit_test.shape)

rede_neural_credit = MLPClassifier(max_iter= 1000, verbose= True, tol= 0.0000100,
                                   hidden_layer_sizes= (20,100))
rede_neural_credit.fit(X_credit_train, Y_credit_train)

previsoes = rede_neural_credit.predict(X_credit_test)
# print(previsoes)

print(accuracy_score(Y_credit_test, previsoes))
