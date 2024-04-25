import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from scipy.stats import sem
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def gakusyuu(filename):
    # csvファイルの読み込み
    df = pd.read_csv(filename).values
    npArray1 = df[:,1:]
    retu = npArray1[0,:].size
    # print(retu)

    # 説明変数の格納
    x = npArray1[:, 0:retu-1]
    # print(x.shape)

    # 正規化------------------------
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # ------------------

    # 目的変数の格納
    y = npArray1[:, retu-1:retu].flatten()
    # print(y.shape)

    ave = np.zeros((2,2))

    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(x, y):
        # どう分割されたか確認する
        # print('TRAIN:', y[train_index], 'TEST:', y[test_index])
        # print(train_index)
        # print(test_index)

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(x[train_index],y[train_index])

        pred_test = clf.predict(x[test_index])
        f1score_test = f1_score(y[test_index], pred_test)
        cm = confusion_matrix(y[test_index], pred_test)

        if (f1score_test>f1score):
            with open('model/model_kNN.pickle', mode='wb') as f:
                pickle.dump(clf,f,protocol=2)
            f1score = f1score_test

        # print(cm)
        # print('テストデータに対するF値： %.2f' % f1score_test)
        ave = ave + cm

    print(ave)

gakusyuu('tokuchouryou/toku_conv.csv')
