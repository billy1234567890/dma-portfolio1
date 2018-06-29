import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
#preprocessing data
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2, columns_to_ignore=[2,7])
data2 = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)

for p in data:
    if p[1] == 'female':
        p[1] = 1
    else:
        p[1] = 0

net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation = 'softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=100, batch_size=16, show_metric=True)

dicaprio = [3, 'Jack Dawson', 'male', 19, 0,0,'N/A', 5.000]
print (model.predict([[2, 0, 14, 0, 0, 33]]))

sum=0

for row in data:
    sum += float(row[5])
print sum/len(data)



