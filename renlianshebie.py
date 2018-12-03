# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 15:51:14 2016

@author: JiaY
"""
from time import time
from PIL import Image
import glob
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#设置解释器为utf8编码，不知为何文件开头的注释没用。
#尽管这样设置，在IPython下仍然会出错，只能用原装Python解释器执行本程序
#reload(sys)
#sys.setdefaultencoding("utf8")
#print (sys.getdefaultencoding())

PICTURE_PATH = u"C:\\Users\\76947\\Desktop\\machine learning\\databat"

all_data_set = [] #原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = [] #总数据对应的类标签

def get_picture():
    label = 1
    #读取所有图片并一维化
    while (label <= 20):
        for name in glob.glob(PICTURE_PATH + "\\s" + str(label) + "\\*.pgm"):
            img = Image.open(name)
            #img.getdata()
            #np.array(img).reshape(1, 92*112)
            all_data_set.append( list(img.getdata()) )
            all_data_label.append(label)
        label += 1

get_picture()

n_components = 16#这个降维后的特征值个数如果太大，比如100，结果将极其不准确，为何？？
pca = PCA(n_components = n_components, svd_solver='auto', 
          whiten=True).fit(all_data_set)
#PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
#X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
y = np.array(all_data_label)


#输入核函数名称和参数gamma值，返回SVM训练十折交叉验证的准确率
def SVM(kernel_name, param):
    #十折交叉验证计算出平均准确率
    #n_splits交叉验证，随机取
    kf = KFold(n_splits=10, shuffle = True)
    precision_average = 0.0
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}#自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced', gamma = param),
                       param_grid)
    for train, test in kf.split(X):
        clf = clf.fit(X[train], y[train])
        #print(clf.best_estimator_)
        test_pred = clf.predict(X[test])
        #print classification_report(y[test], test_pred)
        #计算平均准确率
        precision = 0
        for i in range(0, len(y[test])):
            if (y[test][i] == test_pred[i]):
                precision = precision + 1
        precision_average = precision_average + float(precision)/len(y[test])
    precision_average = precision_average / 10    
    #print (u"准确率为" + str(precision_average))
    return precision_average

t0 = time()    
kernel_to_test = ['rbf', 'poly', 'sigmoid']
#rint SVM(kernel_to_test[0], 0.1)
plt.figure(1)

for kernel_name in kernel_to_test:
    x_label = np.linspace(0.0001, 1, 100)
    y_label = []
    for i in x_label:
        y_label.append(SVM(kernel_name, i))
    plt.plot(x_label, y_label, label=kernel_name)
    
         
print("done in %0.3fs" % (time() - t0))    
plt.xlabel("Gamma")
plt.ylabel("Precision")
plt.title('Different Kernels Contrust') 
plt.legend()
plt.show()    
    
    
    
"""
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)
clf = clf.fit(X_train, y_train)
test_pred = clf.predict(X_test)
print classification_report(y_test, test_pred)

#十折交叉验证计算出平均准确率
precision_average = 0.0
for train, test in kf.split(X):
    clf = clf.fit(X[train], y[train])
    #print(clf.best_estimator_)
    test_pred = clf.predict(X[test])
    #print classification_report(y[test], test_pred)
    #计算平均准确率
    precision = 0
    for i in range(0, len(y[test])):
        if (y[test][i] == test_pred[i]):
            precision = precision + 1
    precision_average = precision_average + float(precision)/len(y[test])
precision_average = precision_average / 10    
print ("准确率为" + str(precision_average))
print("done in %0.3fs" % (time() - t0))
"""
"""               
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(all_data_pca, all_data_label)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
all_data_set_pred = clf.predict(all_data_pca)
#target_names = range(1, 11)
print(classification_report(all_data_set_pred, all_data_label))
"""
