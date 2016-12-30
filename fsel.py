import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('TkAgg')
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

# import and look at centers data
datafile="centers_data_wvorp.csv"

with open("centers_data_wvorp.csv", 'r') as myFile:
    dataLines = myFile.readlines()

data_temp = []
for z in range(1, len(dataLines)):
    data_temp.append(dataLines[z].split(','))
    # print data_temp[x-1]

data = []
for i in range(len(data_temp)):
    temp = []
    for j in range(1, len(data_temp[0])):
        if data_temp[i][j] == '':
            temp.append(0)
        else:
            temp.append(float(data_temp[i][j]))
    temp.append(str(data_temp[i][0]))

    data.append(temp)

# scale data
train = data
temp = np.array(data)[:, :-1]
scaler = preprocessing.StandardScaler().fit(temp[:, 0:-1])
centers = scaler.transform(temp[:, 0:-1]).tolist()
centers = np.array(centers)

print type(centers)

corrected_temp = []
for n in temp:
    n = [float(x) for x in n]
    corrected_temp.append(n)

corrected_temp = np.array(corrected_temp)

X = corrected_temp[:, :-1].tolist()
y = corrected_temp[:, -1].tolist()
print X
print y

# change N_feat parameter below for identifying the N most important features that correlate best to VORP
# Top 4 are DBPM, PTS, BLK, STL which makes sense for centers
estimator = SVR(kernel="linear")
N_feat = 4
selector = RFE(estimator, N_feat, step=0.1)
selector = selector.fit(X, y)
print selector.support_
print selector.ranking_

label = dataLines[0].split(',')
label.remove('Player')
label.remove('VORP\r\n')

print "\n"
print "Top", N_feat, "Features:"
for n in range(len(selector.ranking_)):
    if selector.ranking_[n] == 1:
        print label[n]
