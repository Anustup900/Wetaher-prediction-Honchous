# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv(r"C:\Users\Anustup\Desktop\weather.csv")

dataset['temperature'].fillna(0, inplace=True)

dataset['humidity'].fillna(dataset['humidity'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['temperature'] = X['temperature'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X,y)
# Saving model to disk
pickle.dump(decision_tree, open('weather.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('weather.pkl','rb'))
print(model.predict([[2, 9, 6]]))