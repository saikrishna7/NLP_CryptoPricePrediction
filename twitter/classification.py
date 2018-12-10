from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np
from time import time
import random
import pandas as pd
seed = 666
random.seed(seed)

class Classification:

    def test_classifier(X_train, y_train, X_test, y_test, classifier):
        # print("train data head: ", X_train.head())
        print("train labels head: ", y_train.head())
        print("train data shape: ", X_train.shape)
        # print("test data head: ", X_test.head())
        print("test labels head: ", y_test.head())
        print("test data shape: ", X_test.shape)


        print("")
        print("===============================================")
        classifier_name = str(type(classifier).__name__)
        print("Testing " + classifier_name)

        now = time()
        list_of_labels = sorted(list(set(y_train)))
        print("list of labels: ",list_of_labels)
        model = classifier.fit(X_train, y_train)
        print("Learing time {0}s".format(time() - now))
        now = time()
        predictions = model.predict(X_test)
        print("Predicting time {0}s".format(time() - now))

        precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
        print("=================== Results ===================")
        print("            Negative     Neutral     Positive")
        print("F1       " + str(f1))
        print("Precision" + str(precision))
        print("Recall   " + str(recall))
        print("Accuracy " + str(accuracy))
        print("===============================================")

        return precision, recall, accuracy, f1

    def cv(classifier, X_train, y_train):
        print("===============================================")
        classifier_name = str(type(classifier).__name__)
        now = time()
        print("Crossvalidating " + classifier_name + "...")
        accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
        print("Crosvalidation completed in {0}s".format(time() - now))
        print("Accuracy: " + str(accuracy[0]))
        print("Average accuracy: " + str(np.array(accuracy[0]).mean()))
        print("===============================================")
        return accuracy

