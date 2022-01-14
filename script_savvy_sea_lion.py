"""
the classification/regression part is partially quoted from the source
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
with regards and thanks to the tutorial of usage of sklearning package posts
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# random seed format
np.random.seed(2022)

transactions = pd.read_csv("transactions.csv")
customers = pd.read_csv("customers.csv")
geo = pd.read_csv("geo.csv")

# data preparation for OFFER_STATUS
transactions["OFFER_STATUS"] = transactions["OFFER_STATUS"].replace(["LOST", "Lost", "LOsT", "Lose"], 0.)
transactions["OFFER_STATUS"] = transactions["OFFER_STATUS"].replace(["WIN", "Win", "Won", "WON"], 1.)

# converting the date
transactions["MO_CREATED_DATE"] = pd.to_datetime(transactions["MO_CREATED_DATE"], errors='ignore')
transactions["MO_CREATED_DATE"] = pd.to_datetime(transactions["MO_CREATED_DATE"], errors='ignore')

transactions["SO_CREATED_DATE"] = pd.to_datetime(transactions["SO_CREATED_DATE"], errors='ignore')
transactions["SO_CREATED_DATE"] = pd.to_datetime(transactions["SO_CREATED_DATE"], errors='ignore')

# finding valid data for the df
df = transactions.drop(columns=["MO_ID", "SO_ID"])

# dealing with end customers
df["END_CUSTOMER"] = df["END_CUSTOMER"].fillna(-1).replace({"No": 0, "Yes": 1}).astype(int)

# dealing with isic
df["ISIC"] = df["ISIC"].fillna(0)


# converting customer ids to nums
def remove_quote(entry):
    return entry[1:-1]


df["CUSTOMER"] = df["CUSTOMER"].map(remove_quote, 'ignore')

# TODO: join and groupby country dealing with the customer ids

# turning high/low margin products to pertage and comparable
for letter in ["A", "B", "C", "D", "E"]:
    df["Percentage_of_Product_" + letter] = df["COSTS_PRODUCT_" + letter] / (df["OFFER_PRICE"])
    df = df.drop(columns=["COSTS_PRODUCT_" + letter])

# one-hot encoding for nominals
df = pd.get_dummies(df, columns=["TECH", "BUSINESS_TYPE", "PRICE_LIST", "SALES_LOCATION"])
df = df.drop(columns=["CUSTOMER","MO_CREATED_DATE","SO_CREATED_DATE","OFFER_TYPE","TEST_SET_ID"])


# Modeling the data

# dividing the outcomes and variables
df = df[df["OFFER_STATUS"].notna()]
Y = df["OFFER_STATUS"]
X = df.drop(columns="OFFER_STATUS")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# the scaling matter, for good habits
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training and fit the best tree in the model
classifier = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample")
# classifier = VotingClassifier()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score,balanced_accuracy_score

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(balanced_accuracy_score(Y_test, Y_pred))
print(precision_score(Y_test, Y_pred))
print(recall_score(Y_test, Y_pred))
