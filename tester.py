"""
the classification/regression part is partially quoted from the source
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
with regards and thanks to the tutorial of usage of sklearning package posts
"""
# Loading packages and data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set a fixed random seed to make the work reproducible
np.random.seed(2022)

transactions = pd.read_csv("transactions.csv")
customers = pd.read_csv("customers.csv")
geo = pd.read_csv("geo.csv")

# Join transactions with geo, customers
# We use the left outer join to ensure there is no data loss in transactions
df = transactions.merge(geo, how="left")


# The data type of attribute "CUSTOMER" in data transactions is String
# We need to firstly convert the data type into int

# Converting string to nums
def remove_quote(entry):
    return entry[1:-1]


def remove_NV(entry):
    if entry in ['#NV', 'NA', np.nan]:
        return np.nan
    else:
        return float(entry)


df["CUSTOMER"] = df["CUSTOMER"].map(remove_quote, 'ignore')
df["CUSTOMER"] = df["CUSTOMER"].map(remove_NV).dropna().astype(int)
customers = customers.dropna()

# The data type of "REV_CURRENT_YEAR" is string
# We need to convert the data type into float
customers["REV_CURRENT_YEAR"] = customers["REV_CURRENT_YEAR"].map(remove_quote).astype(float)

# The format of attribute "COUNTRY" is different in data customers and transactions
# We need to firstly convert them to the same format
df["COUNTRY"] = df["COUNTRY"].map({"CH": "Switzerland", "FR": "France"})
df = df.merge(customers, how="left", on=["CUSTOMER", "COUNTRY"])

# Exploratory Analysis and Preprocessing

# data preparation for OFFER_STATUS
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["LOST", "Lost", "LOsT", "Lose"], 0.)
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["WIN", "Win", "Won", "WON"], 1.)

df = df[df["CUSTOMER"].notna()]
df["ISIC"] = df["ISIC"].fillna(df["ISIC"].mean())

# Dealing with the dates
df["CREATION_YEAR"] = pd.to_datetime(df["CREATION_YEAR"]).dt.year
df["MO_CREATED_DATE"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.month
df["SO_CREATED_DATE"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.month

df["CREATION_YEAR"] = df["CREATION_YEAR"].fillna(df["CREATION_YEAR"].mean())

# Uniting currency with cny
df["CURRENCY"] = df["CURRENCY"].map({"Chinese Yuan": 1, "Euro": 7.25, "US Dollar": 6.36, "Pound Sterling": 8.72, 0: 0})
for col in ["CURRENCY", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2"]:
    df[col] = df[col].fillna(df[col].mean())

df["REV_CURRENT_YEAR"] = df["REV_CURRENT_YEAR"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.1"] = df["REV_CURRENT_YEAR.1"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.2"] = df["REV_CURRENT_YEAR.2"] * df["CURRENCY"]

df["COUNTRY"] = df["COUNTRY"].fillna("UNKNOWN")
df["OWNERSHIP"] = df["OWNERSHIP"].fillna("UNKNOWN")


# Process the end_customers
def map_end_customer(entry):
    if entry in ["No", np.nan]:
        return 0
    elif entry in ["Yes"]:
        return 1
    else:
        return 1


df["END_CUSTOMER"] = df["END_CUSTOMER"].map(map_end_customer)

# Turning high/low margin products to percentage and make them comparable(effective)
for letter in ["A", "B", "C", "D", "E"]:
    df["Percentage_of_Product_" + letter] = df["COSTS_PRODUCT_" + letter] / (df["OFFER_PRICE"])
    df = df.drop(columns=["COSTS_PRODUCT_" + letter])

# One-hot encoding for nominal
df = pd.get_dummies(df, columns=["TECH", "BUSINESS_TYPE", "PRICE_LIST", "OWNERSHIP", "OFFER_TYPE", "COUNTRY",
                                 "SALES_BRANCH"])

# Drop useless variables
df = df.drop(columns=["MO_ID", "SO_ID", "SALES_LOCATION",
                      "SALES_OFFICE", "CURRENCY", "CUSTOMER", "TEST_SET_ID"])

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

# test part
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))
