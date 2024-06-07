from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor

from sklearn.svm import SVR
from sklearn.svm import SVC 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

import matplotlib.pyplot as plt


# --------------------------------------
# ------- knn_plot()
# ---------- KNN Plot to select best n_neighbors value
# --------------------------------------
def knn_plot( data ):
    X_train, X_test, y_train, y_test = data
    k_range = 10
    train_scores = []
    test_scores = []
    for k in range(1, k_range, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        test_score = knn.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
        
    # Plot the results
    plt.plot(range(1, k_range, 2), train_scores, marker='o', label="training scores")
    plt.plot(range(1, k_range, 2), test_scores, marker="x", label="testing scores")
    plt.xlabel("k neighbors")
    plt.ylabel("accuracy score")
    plt.legend()
    plt.show()


# --------------------------------------
# ------- Test_model()
# ---------- 
# --------------------------------------
def test_model(model, data):
    X_train, X_test, y_train, y_test = data
    reg = model.fit(X_train, y_train)
    print(f'Model: {type(reg).__name__}')
    print(f'Train score: {reg.score(X_train, y_train)}')
    print(f'Test Score: {reg.score(X_test, y_test)}\n')
    # Add balancedscore
    # Add Classification Report
    plt.show()


# --------------------------------------
# ------- run_classification_models(data, k_value)
# ---------- 
# --------------------------------------
def run_classification_models(data, k_value):
    print(f"\n----------------------------------------------------------------")
    print(f"-------- Classification Models")
    print(f"----------------------------------------------------------------")

    test_model(SVC(kernel='linear'), data)
    test_model(KNeighborsClassifier(n_neighbors=k_value), data)
    test_model(tree.DecisionTreeClassifier(), data)
    test_model(RandomForestClassifier(), data)
    test_model(ExtraTreesClassifier(random_state=1), data)
    test_model(GradientBoostingClassifier(random_state=1), data)
    test_model(AdaBoostClassifier(random_state=1), data)
    test_model(LogisticRegression(), data)


# --------------------------------------
# ------- run_regression_models(data, k_value)
# ---------- 
# --------------------------------------
def run_regression_models(data, k_value):
    print(f"\n----------------------------------------------------------------")
    print(f"-------- Regression Models")
    print(f"----------------------------------------------------------------")

    test_model(LinearRegression(), data)
    test_model(KNeighborsRegressor(), data)
    test_model(RandomForestRegressor(), data)
    test_model(ExtraTreesRegressor(), data)
    test_model(AdaBoostRegressor(), data)
    test_model(SVR(C=1.0, epsilon=0.2), data)


# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------
