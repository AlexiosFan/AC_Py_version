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

df = transactions.merge(geo, how="left")
df["COUNTRY"] = df["COUNTRY"].map({"CH":"Switzerland", "FR":"France"})

# data preparation for OFFER_STATUS
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["LOST", "Lost", "LOsT", "Lose"], 0.)
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["WIN", "Win", "Won", "WON"], 1.)

# converting the date
df["MO_CREATED_DATE"] = pd.to_datetime(df["MO_CREATED_DATE"], errors='ignore')
df["MO_CREATED_DATE"] = pd.to_datetime(df["MO_CREATED_DATE"], errors='ignore')

df["SO_CREATED_DATE"] = pd.to_datetime(df["SO_CREATED_DATE"], errors='ignore')
df["SO_CREATED_DATE"] = pd.to_datetime(df["SO_CREATED_DATE"], errors='ignore')

# finding valid data for the df
df = df.drop(columns=["MO_ID", "SO_ID", "END_CUSTOMER", "ISIC"])

# converting customer ids to nums
def remove_quote(entry):
    return entry[1:-1]

#merging 3 lists
df["CUSTOMER"] = df["CUSTOMER"].map(remove_quote, 'ignore')

def remove_NV(entry):
    if entry in ['#NV', 'NA', np.nan] :
        return np.nan
    else:
        return float(entry)

df["CUSTOMER"] = df["CUSTOMER"].map(remove_NV).dropna().astype(int)
customers = customers.dropna()
customers["REV_CURRENT_YEAR"] = customers["REV_CURRENT_YEAR"].map(remove_quote).astype(float)

# left join a second list
df = df.merge(customers, how="left", on=["CUSTOMER", "COUNTRY"])
df = df[df["CUSTOMER"].notna()]
df = df[df["CREATION_YEAR"].notna()]
df = df.drop(columns=["SALES_LOCATION", "SALES_OFFICE", "SALES_BRANCH", "CREATION_YEAR"])

# uniting currency with cny
df["CURRENCY"] = df["CURRENCY"].map({"Chinese Yuan": 1, "Euro": 7.25, "US Dollar": 6.36, "Pound Sterling": 8.72})
df["REV_CURRENT_YEAR"] = df["REV_CURRENT_YEAR"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.1"] = df["REV_CURRENT_YEAR.1"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.2"] = df["REV_CURRENT_YEAR.2"] * df["CURRENCY"]
df.drop("CURRENCY", 1)


# turning high/low margin products to pertage and comparable

for letter in ["A", "B", "C", "D", "E"]:
    df["Percentage_of_Product_" + letter] = df["COSTS_PRODUCT_" + letter] / (df["OFFER_PRICE"])
    df = df.drop(columns=["COSTS_PRODUCT_" + letter])


# one-hot encoding for nominals
df = pd.get_dummies(df, columns=["TECH", "BUSINESS_TYPE", "PRICE_LIST", "COUNTRY", "OWNERSHIP"])


# TODO: a couple of variables deleted in the current model, to be handled or deleted
df = df.drop("CUSTOMER", 1).drop("MO_CREATED_DATE", 1).drop("SO_CREATED_DATE", 1).drop("OFFER_TYPE", 1).drop("TEST_SET_ID", 1)

df = df[df["OFFER_STATUS"].notna()]

Y = df["OFFER_STATUS"]
X = df.drop("OFFER_STATUS", 1)

# dividing the outcomes and variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# the scaling matter, for good habits
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training and fit the best tree in the model
classifier = RandomForestClassifier(n_estimators=30, random_state=0, criterion="entropy", max_features="sqrt")
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# test part
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))