# -*- coding: utf-8 -*-
"""
Created on 2017/5/4

@author: Naiive
定义向量机
"""
__all__ = ["SVM1", "SVM2"]
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
#定义SVM支持向量机,返回预测和准确率
def SVM1(kernel_name, param, pic):
    x_train, x_test, y_train, y_test = pic.get_data()
    target_n = pic.get_target_n()
    n_classes = pic.get_nclasses()
    precision_average = 0.0

    #构造最优向量机
    print("正在进行训练...")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma' : param,}#自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced'),
                       param_grid)
    clf = clf.fit(x_train, y_train)
    print("完成，共计 %0.3f秒" % (time() - t0))
    print("最优分类向量机找到:")
    print(clf.best_estimator_)

    return clf
def SVM2(kernel_name, param, pic):
    x_train, x_test, y_train, y_test = pic.get_data()
    target_n = pic.get_target_n()
    n_classes = pic.get_nclasses()
    precision_average = 0.0
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma' : param,}#自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced'),
                       param_grid)
    clf = clf.fit(x_train, y_train)


    #预测图片
    t0 = time()
    test_pred = clf.predict(x_test)
    prediction = 0
    for i in range(len(test_pred)):
    	if test_pred[i] == y_test[i]:
    		prediction += 1
    precision_average = prediction / len(y_test)
    return test_pred, precision_average