import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# print train.head()

#1 - understanding the data
print('Total number of passangers in the training data...', len(train))
print('Number of passangers in the training data who survived...', len(train[train['Survived'] == 1]))

print
print('% of men who survived', 100*np.mean(train['Survived'][train['Sex'] == 'male']))
print('% of women who survived', 100*np.mean(train['Survived'][train['Sex'] == 'female']))

print
print('% of men who survived', 100*np.mean(train['Survived'][train['Age'] < 10]))
print('% of women who survived', 100*np.mean(train['Survived'][train['Age'] > 10]))

print
print('% of men who survived', 100*np.mean(train['Survived'][train['Pclass'] == 1]))
print('% of women who survived', 100*np.mean(train['Survived'][train['Pclass'] == 2]))
print('% of women who survived', 100*np.mean(train['Survived'][train['Pclass'] == 3]))


#2 - Data preprocessing
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

#3- Missing Values
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))
test['Age'] = test['Age'].fillna(np.mean(test['Age']))
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

#4- Feature Selection
train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]
test2 = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]

#5- Separate features and labels
X = train.drop('Survived', axis = 1)
y = train['Survived']

#6- Split training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#7- Make the actual prediction
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth = 3)
classifier.fit(X_train, y_train)

#8- evaluate the model
from sklearn.metrics import accuracy_score
print
print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))
print('Validation accuracy', accuracy_score(y_test, classifier.predict(X_test)))

print
print 
predict = classifier.predict(test2)
test['Survived'] = predict
test = test[['PassengerId', 'Survived']]
test.to_csv(path_or_buf='output.csv', index=False)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)