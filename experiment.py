import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits,load_iris
from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import train_test_split


digits = load_digits()
iris = load_iris()

def create_set(name, show=True):
    X = name.data
    y = name.target

    svm_rbf_accuracy_list = []
    nb_accuracy_list = []
    dt_accuracy_list = []
    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

        # Creating rbf kernel svm 
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        rbf_acc = accuracy_score(y_pred, y_test)
        svm_rbf_accuracy_list.append(rbf_acc)

        # Creating a naive bayes classifier
        gnb = GaussianNB()
        y_nb_pred = gnb.fit(X_train, y_train).predict(X_test)
        nb_acc = accuracy_score(y_nb_pred, y_test)
        nb_accuracy_list.append(nb_acc)

        # Creating a Decision Tree Classifier
        clf = DecisionTreeClassifier(max_depth=10)
        y_dt_pred = clf.fit(X_train, y_train).predict(X_test)
        dt_acc = accuracy_score(y_dt_pred, y_test)
        dt_accuracy_list.append(dt_acc)

    svm_rbf_avg_accuracy = np.average(svm_rbf_accuracy_list)
    nb_avg_accuracy = np.average(nb_accuracy_list)
    dt_avg_accuracy = np.average(dt_accuracy_list)

    # Introducing Noise
    svm_rbf_noisy_accuracy_list = []
    nb_noisy_accuracy_list = []

    for i in range(0, 400, 4):
        x_noisy = X + np.random.normal(0, i/10, X.shape)
        X_noisy_train, X_noisy_test, y_noisy_train, y_noisy_test = train_test_split(x_noisy, y, test_size = 0.20)

        # RBF kernel svm
        svc_noisy_lassifier = SVC(kernel='rbf')
        svc_noisy_lassifier.fit(X_noisy_train, y_noisy_train)
        y_noisy_pred = svc_noisy_lassifier.predict(X_noisy_test)
        bf_noisy_acc = accuracy_score(y_noisy_pred, y_noisy_test)
        svm_rbf_noisy_accuracy_list.append(bf_noisy_acc)

        # Naive Bayes
        gnb = GaussianNB()
        y_noisy_nb_pred = gnb.fit(X_noisy_train, y_noisy_train).predict(X_noisy_test)
        nb_noisy_acc = accuracy_score(y_noisy_nb_pred, y_noisy_test)
        nb_noisy_accuracy_list.append(nb_noisy_acc)

    svm_noisy_relative_list = []
    for i in svm_rbf_noisy_accuracy_list:
        m = i/np.max(svm_rbf_noisy_accuracy_list)
        svm_noisy_relative_list.append(m)
    
    nb_noisy_relative_list = []
    for i in nb_noisy_accuracy_list:
        m = i/np.max(nb_noisy_accuracy_list)
        nb_noisy_relative_list.append(m)

    def avg_acc_loss(l):
        acc_loss_list = []
        for i in range(1, len(l)):
            m = l[i] - l[i-1]
            acc_loss_list.append(m)
        lavgpercent = np.average(acc_loss_list)*100
        return round(lavgpercent, 3)

    def calc_dif(l1, l2):
        avg_acc_loss_dif = avg_acc_loss(l1) - avg_acc_loss(l2)
        return round(avg_acc_loss_dif*100, 2)

        
    defference_l = []
    for i in range(100):
        n = svm_noisy_relative_list[i] - nb_noisy_relative_list[i]
        defference_l.append(n)
    defference = np.average(defference_l)

    if show==True:
        print(round(svm_rbf_avg_accuracy, 2))
        print(round(nb_avg_accuracy, 2))
        print(round(dt_avg_accuracy, 2))
        print('Avarage loss of SVM with RBF kernel: ' + avg_acc_loss(svm_noisy_relative_list) + '\nAvarage loss of Gaussian Naive Bayes:' + avg_acc_loss(nb_noisy_relative_list))
        # print(np.average(svm_rbf_noisy_accuracy_list))
        x = np.arange(0,10,0.1)
        plt.plot(x, svm_noisy_relative_list, label='SVM with RBF Kernel')
        plt.plot(x, nb_noisy_relative_list, label='Naive Bayes')
        plt.ylabel('RelativeModel Accuracy')
        plt.xlabel('Standard diviation')
        plt.show()
    else:
        print('done itteration')
        return calc_dif(svm_rbf_noisy_accuracy_list, nb_noisy_accuracy_list)

# create_set(digits)
def make_experiment(dataset, itterations):
    avg_def_list=[]
    for i in range(itterations):
        svt_to_nb_loss_diff = create_set(dataset, show=False)
        avg_def_list.append(svt_to_nb_loss_diff)
    return str(round(np.average(avg_def_list), 2)) + '%'

print(make_experiment(iris, 100))