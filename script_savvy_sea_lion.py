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
df = transactions.drop(columns=["MO_ID", "SO_ID", "TEST_SET_ID"])

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

train_set = df[df["OFFER_STATUS"].notna()]
test_set_init = df[df["OFFER_STATUS"].isna()]

# TODO: a couple of variables deleted in the current model, to be handled or deleted
train_set = train_set.drop("CUSTOMER", 1).drop("MO_CREATED_DATE", 1).drop("SO_CREATED_DATE", 1).drop("OFFER_TYPE", 1)
test_set = test_set_init.drop("CUSTOMER", 1).drop("MO_CREATED_DATE", 1).drop("SO_CREATED_DATE", 1).drop("OFFER_TYPE", 1)


# dividing the outcomes and variables
Y_train = train_set["OFFER_STATUS"].values
X_train = train_set.drop("OFFER_STATUS", 1).values

Y_test = test_set["OFFER_STATUS"].values
X_test = test_set.drop("OFFER_STATUS", 1).values

# the scaling matter, for good habits
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training and fit the best tree in the model
classifier = RandomForestClassifier(n_estimators=20, random_state=0, criterion="entropy", max_features="sqrt")
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# output the results
result = pd.DataFrame([test_set_init["CUSTOMER"].to_numpy(), np.split(Y_pred, len(Y_pred))], index=["id", "prediction"]).T
result["prediction"] = result["prediction"].map(np.sum).astype(int)
result.to_csv("prediction_savvy_sea_lion_1.csv")
