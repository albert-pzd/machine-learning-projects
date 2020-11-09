import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib as mpl

#set picture resolution
mpl.rcParams['figure.dpi'] = 300

"""Logistic Regression"""
data = datasets.load_iris()
X = data.data[:,:2][:100]
Y= data.target[:100]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, 
                                                    random_state =0)
binary_classifier = LogisticRegression()
binary_classifier.fit(X_train, Y_train)

""""Plot"""
plot_decision_regions(
    X_train, Y_train, clf=binary_classifier, legend=2, colors='red,blue'
)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('logistic regression')
plt.show()

print("Accuracy on training set: ", binary_classifier.score(X_train, Y_train))
print("Accuracy on training set: ", binary_classifier.score(X_test, Y_test))

"""SVM"""
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train, Y_train)
sv = svm_clf.support_vectors_

w = svm_clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 8)
yy = a * xx - (svm_clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(svm_clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

""""Plot"""
plot_decision_regions(
    X_train, Y_train, clf=svm_clf, legend=2, colors='red,blue'
)
plt.scatter(sv[:,0], sv[:,1], s=120, facecolors='none', edgecolors='green', linewidths=1)
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM')
plt.show()

print("Accuracy on training set: ", svm_clf.score(X_train, Y_train))
print("Accuracy on test set: ", svm_clf.score(X_test, Y_test))
print('The value of the margin is: ', margin)
print('The weight vector is orthogonal to the decision boundary.')


X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.4, 
                                                    random_state =0)
svm_clf2 = svm.SVC(kernel='linear')
svm_clf2.fit(X_train2, Y_train2)
sv2 = svm_clf2.support_vectors_
plot_decision_regions(
    X_train2, Y_train2, clf=svm_clf2, legend=2, colors='red,blue'
)
plt.scatter(sv2[:,0], sv2[:,1], s=120, facecolors='none', edgecolors='green', linewidths=1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM2')
plt.show()

print("Accuracy on test set: ", svm_clf2.score(X_test2, Y_test2))


"""We apply a kernel function to transform the data into a higher 
dimensional space, so linearly non separable data points become separable."""
X = data.data[:,:2]
Y= data.target
svm_clf3 = svm.SVC(kernel='rbf')
svm_clf3.fit(X, Y)
sv3 = svm_clf3.support_vectors_
plot_decision_regions(
    X, Y, clf=svm_clf3, legend=2, colors='red,blue,green'
)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM3')
plt.show()

    
