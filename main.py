# jjacobi 2017 - Machine Learning
# jjacobi@student.42.fr


# Required libraries

# PANDAS
import pandas
from pandas.plotting import scatter_matrix
# MATPLOTLIB
"""
If you got an error importing tkinter please install python3-tk
Fedora 27:
(➜ sudo dnf install python3-tkinter-3.6.3-2.fc27.x86_64)
"""
import matplotlib.pyplot as plt
# SKLEARN
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


"""
If you don't understand any function dont forget to do help(func_name) in the
python interpreter
Launch in interactive mode with python -i main.py
"""

# Loading the dataset in a dictionary
iris_database = "./iris_database.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(iris_database, names=names)


def multiplot():
    """
        This will print a plot about all the datas we have in entry
        This visual support can be usefull to find data correlations
    """
    scatter_matrix(dataset)
    plt.show()


"""
We will remove validation_size percentage from the dataset to obtain a training
and a validation data set.
Machine learning became strong during the last few years cause of the amount of
data we now have, in many case the validation dataset represent less than 1
percent of the dataset
The currently used dataset only contain 150 values so 1 pourcent or less
wouldn't be accurate so we take 20 pourcent
"""
array = dataset.values
validation_size = 0.20
"""
When we split our dataset it is randomly splited (with np.random) in scipy,
to avoid to have different results each time we launch the program due to
the order of the dataset we use random_state that will define the way the
dataset is splited
Avoid to use this in production
The value will only change the order of the elements in the dataset
"""
seed = 8
X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(array[:, 0:4], array[:, 4],
                                     test_size=validation_size,
                                     random_state=seed)

"""
This is the dictionary of the algos we will compare
"""

models = {
    'LR': LogisticRegression(),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(),
    'NB': GaussianNB(),
    'SVM': SVC()
}

results = []
names = []


def benchalgos(hide=False):
    # We are using the metric accuracy, the result will be in %
    scoring = 'accuracy'
    global results
    global names
    """
     kfold: Split dataset into k consecutive folds
     Each fold is then used once as a validation while the k - 1 remaining
     folds form the training set.
    """
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    for name in models.keys():
        """
            Evaluate the scoring (in this case accuracy) thanks to the selected
            model with the training data and kfold
        """
        cv_results = model_selection.cross_val_score(models[name], X_train,
                                                     Y_train, cv=kfold,
                                                     scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%5s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
        if not (hide):
            print(msg)


def benchplot():
    benchalgos(hide=True)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def test(name):
    global models
    models[name].fit(X_train, Y_train)
    predictions = models[name].predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
