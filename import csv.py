import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
df = df[df["OFFER_STATUS"].notna()]

# dealing with end customers
df["END_CUSTOMER"] = df["END_CUSTOMER"].fillna(-1).replace({"No": 0, "Yes": 1}).astype(int)

# dealing with isic
df["ISIC"] = df["ISIC"].fillna(0)
# converting customer ids to nums
def remove_quote(entry):
    return entry[1:-1]

df["CUSTOMER"] = df["CUSTOMER"].map(remove_quote, 'ignore')


# TODO : join and groupby country dealing with the customer ids

# turning high/low margin products to pertage and comparable
for letter in ["A", "B", "C", "D", "E"]:
    df["Percentage_of_Product_"+letter] = df["COSTS_PRODUCT_"+letter]/(df["MATERIAL_COST"]+df["SERVICE_COST"])
    df = df.drop(columns=["COSTS_PRODUCT_"+letter])

# one-hot encoding for nominals
df = pd.get_dummies(df, columns=["TECH", "OFFER_TYPE", "BUSINESS_TYPE", "PRICE_LIST"])
df = df.drop("CUSTOMER", 1).drop("MO_CREATED_DATE", 1).drop("SO_CREATED_DATE", 1).drop("SALES_LOCATION", 1)

# dividing the outcomes and variables
Y = df["OFFER_STATUS"].values
X = df.drop("OFFER_STATUS", 1).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# the scaling matter, for good habits
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))














