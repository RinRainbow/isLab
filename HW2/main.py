import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)
X_train_fpath = './HW2/data/X_train'
Y_train_fpath = './HW2/data/Y_train'
X_test_fpath = './HW2/data/X_test'
#output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = int)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

'''
print(X_train)
print(Y_train)
print(X_test)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
'''

# Select Algorithm
clf = RandomForestClassifier()

# Learn
clf.fit(X_train, Y_train)

# Test
Y_test = clf.predict(X_test)

# DataType float -> int
Y_test = Y_test.astype(np.int64)

# Show result
#print("X_test:\n{0},\nY_test:\n{1}".format(X_test,Y_test))

#print(Y_test)


import csv
with open('./HW2/output.csv', mode='w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    header = ['id', 'label']
    print(header)
    csv_writer.writerow(header)
    for i in range(27622):
        row = [str(i), Y_test[i]]
        csv_writer.writerow(row)
        print(row)
