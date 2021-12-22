import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits,load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Loading Data Sets
digits = load_digits()
iris = load_iris()

def create_set(dsetname, show=True):
    X = dsetname.data
    y = dsetname.target

    svm_rbf_accuracy_list = []
    nb_accuracy_list = []
    dt_accuracy_list = []
    knn_accuracy_list = []
    
    if show == True:
        for i in range(100):
            # Creating Dataset
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

            # Creating Knn Classifier
            knn = KNeighborsClassifier(n_neighbors=5)
            y_knn_pred = knn.fit(X_train, y_train).predict(X_test)
            knn_acc = accuracy_score(y_knn_pred, y_test)
            knn_accuracy_list.append(knn_acc)

        svm_rbf_avg_accuracy = np.average(svm_rbf_accuracy_list)
        nb_avg_accuracy = np.average(nb_accuracy_list)
        dt_avg_accuracy = np.average(dt_accuracy_list)
        knn_avg_accuracy = np.average(knn_accuracy_list)

    # Introducing Noise
    svm_rbf_noisy_accuracy_list = []
    nb_noisy_accuracy_list = []
    dt_noisy_accuracy_list = []
    knn_noisy_accuracy_list = []

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

            # Creating a Decision Tree Classifier
        clf = DecisionTreeClassifier(max_depth=10)
        y_noisy_dt_pred = clf.fit(X_noisy_train, y_noisy_train).predict(X_noisy_test)
        dt_noisy_acc = accuracy_score(y_noisy_dt_pred, y_noisy_test)
        dt_noisy_accuracy_list.append(dt_noisy_acc)
 
        # Knn Classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        y_noisy_knn_pred = knn.fit(X_noisy_train, y_noisy_train).predict(X_noisy_test)
        knn_noisy_acc = accuracy_score(y_noisy_knn_pred, y_noisy_test)
        knn_noisy_accuracy_list.append(knn_noisy_acc)

    # Creating Relative Loss Lists
    svm_noisy_relative_list = []
    for i in svm_rbf_noisy_accuracy_list:
        m = i/np.max(svm_rbf_noisy_accuracy_list)
        svm_noisy_relative_list.append(m)
    
    nb_noisy_relative_list = []
    for i in nb_noisy_accuracy_list:
        m = i/np.max(nb_noisy_accuracy_list)
        nb_noisy_relative_list.append(m)

    dt_noisy_relative_list = []
    for i in dt_noisy_accuracy_list:
        m = i/np.max(dt_noisy_accuracy_list)
        dt_noisy_relative_list.append(m)
    
    knn_noisy_relative_list = []
    for i in knn_noisy_accuracy_list:
        m = i/np.max(knn_noisy_accuracy_list)
        knn_noisy_relative_list.append(m)

    def avg_acc_loss(l):
        total = 0
        for i in range(1, len(l)):
            total += (l[i-1] - l[i]) * 100
        return round((total/len(l)), 3)
        
    defference_l = []
    for i in range(100):
        n = svm_noisy_relative_list[i] - nb_noisy_relative_list[i]
        defference_l.append(n)
    defference = np.average(defference_l)

    if show==True:
        # Printing Results
        print('RBF SVM initial accuracy: ' + str(round(svm_rbf_avg_accuracy, 4)*100) + '%')
        print('Naive Bayes initial accuracy: ' + str(round(nb_avg_accuracy, 4)*100) + '%')
        print('Decision Tree initial accuracy: ' + str(round(dt_avg_accuracy, 4)*100) + '%')
        print('k-NN initial accuracy: ' + str(round(knn_avg_accuracy, 4)*100) + '%')
        print('Avarage accuracy loss of SVM with RBF kernel: ' + str(avg_acc_loss(svm_noisy_relative_list)) + '%' +
        '\nAvarage accuracy loss of Gaussian Naive Bayes: ' + str(avg_acc_loss(nb_noisy_relative_list)) + '%' +
        '\nAvarage accuracy loss of Decision Trees: ' + str(avg_acc_loss(dt_noisy_relative_list)) + '%' +
        '\nAvarage accuracyloss of k-Nearest Neighbors: ' + str(avg_acc_loss(knn_noisy_relative_list)) + '%' )

        # Creating the plot
        x = np.arange(0,10,0.1)
        SVMline, = plt.plot(x, svm_noisy_relative_list, label='SVM with RBF Kernel')
        NBline, = plt.plot(x, nb_noisy_relative_list, label='Naive Bayes')
        DTline, = plt.plot(x, dt_noisy_relative_list, label='Decision Trees')
        KNNline, = plt.plot(x, knn_noisy_relative_list, label='k-NN')
        plt.ylabel('Relative Model Accuracy')
        plt.xlabel('Standard deviation')
        plt.legend(handles=[SVMline, NBline, DTline, KNNline])
        # plt.title(label = 'Model Performance with noisy Iris Plants data set')
        plt.title(label = 'Relative Model Performance with noisy Handwritten Digits data set')
        plt.show()

    else:
        return avg_acc_loss(svm_noisy_relative_list), avg_acc_loss(nb_noisy_relative_list), avg_acc_loss(dt_noisy_relative_list), avg_acc_loss(knn_noisy_relative_list)

def make_experiment(dataset, itterations):
    svm_avg_l=[]
    nb_avg_l=[]
    dt_avg_l=[]
    knn_avg_l=[]
    for i in range(itterations):
        svm_avg, nb_avg, dt_avg, knn_avg = create_set(dataset, show=False)
        svm_avg_l.append(svm_avg)
        nb_avg_l.append(nb_avg)
        dt_avg_l.append(dt_avg)
        knn_avg_l.append(knn_avg)
        print('Itteration no {} done'.format(i+1))
    return str(round(np.average(svm_avg_l), 2)) + '% \n' + str(round(np.average(nb_avg_l), 2))+ '% \n' + str(round(np.average(dt_avg_l), 2)) + '% \n'  + str(round(np.average(knn_avg_l), 2)) + '%'


# create_set(iris)
create_set(digits)


# print(make_experiment(digits, 10))

# print(make_experiment(iris, 10))

# print(make_experiment(iris, 20))