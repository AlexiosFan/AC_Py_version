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
df["COUNTRY"] = df["COUNTRY"].map({"CH": "Switzerland", "FR": "France"})

# data preparation for OFFER_STATUS
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["LOST", "Lost", "LOsT", "Lose"], 0.)
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["WIN", "Win", "Won", "WON"], 1.)

# converting the date
df["MO_CREATED_DATE"] = pd.to_datetime(df["MO_CREATED_DATE"], errors='ignore')
df["SO_CREATED_DATE"] = pd.to_datetime(df["SO_CREATED_DATE"], errors='ignore')


# converting customer ids to nums
def remove_quote(entry):
    return entry[1:-1]


# merging 3 lists
df["CUSTOMER"] = df["CUSTOMER"].map(remove_quote, 'ignore')


def remove_NV(entry):
    if entry in ['#NV', 'NA', np.nan]:
        return np.nan
    else:
        return float(entry)


df["CUSTOMER"] = df["CUSTOMER"].map(remove_NV).dropna().astype(int)
customers = customers.dropna()
customers["REV_CURRENT_YEAR"] = customers["REV_CURRENT_YEAR"].map(remove_quote).astype(float)

# left join a second list
df = df.merge(customers, how="left", on=["CUSTOMER", "COUNTRY"])
df = df[df["CUSTOMER"].notna()]
df["ISIC"] = df["ISIC"].fillna(df["ISIC"].mean())

# dealing with the dates
df["CREATION_YEAR"] = pd.to_datetime(df["CREATION_YEAR"]).dt.year
df["MO_CREATED_DATE"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.month
df["SO_CREATED_DATE"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.month

df["CREATION_YEAR"] = df["CREATION_YEAR"].fillna(df["CREATION_YEAR"].mean())

df["CURRENCY"] = df["CURRENCY"].map({"Chinese Yuan": 1, "Euro": 7.25, "US Dollar": 6.36, "Pound Sterling": 8.72, 0: 0})
for col in ["CURRENCY", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2"]:
    df[col] = df[col].fillna(df[col].mean())

df["COUNTRY"] = df["COUNTRY"].fillna("UNKNOWN")
df["OWNERSHIP"] = df["OWNERSHIP"].fillna("UNKNOWN")

# useless variables
df = df.drop(columns=["MO_ID", "SO_ID", "SALES_LOCATION",
             "SALES_OFFICE"])

# end customers
def map_end_customer(entry):
    if entry in ["No", np.nan]:
        return 0
    elif entry in ["Yes"]:
        return 1
    else:
        return 1

df["END_CUSTOMER"] = df["END_CUSTOMER"].map(map_end_customer)

# uniting currency with cny

df["REV_CURRENT_YEAR"] = df["REV_CURRENT_YEAR"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.1"] = df["REV_CURRENT_YEAR.1"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.2"] = df["REV_CURRENT_YEAR.2"] * df["CURRENCY"]
df = df.drop("CURRENCY", 1)
# turning high/low margin products to pertage and comparable(effective)
for letter in ["A", "B", "C", "D", "E"]:
    df["Percentage_of_Product_" + letter] = df["COSTS_PRODUCT_" + letter] / (df["OFFER_PRICE"])
    df = df.drop(columns=["COSTS_PRODUCT_" + letter])


# one-hot encoding for nominal
df = pd.get_dummies(df, columns=["TECH", "BUSINESS_TYPE", "PRICE_LIST", "OWNERSHIP", "OFFER_TYPE", "COUNTRY","SALES_BRANCH"])
# TODO: a couple of variables deleted in the current model, to be handled or deleted
df = df.drop(columns=["CUSTOMER"])

train_set = df[df["OFFER_STATUS"].notna()]
test_set_init = df[df["OFFER_STATUS"].isna()]

train_set = train_set.drop("TEST_SET_ID", 1)
test_set = test_set_init.drop("TEST_SET_ID", 1)

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
classifier = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample")
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# output the results
result = pd.DataFrame([test_set_init["TEST_SET_ID"].to_numpy(), np.split(Y_pred, len(Y_pred))], index=["id", "prediction"]).T
result["prediction"] = result["prediction"].map(np.sum)
result = result.astype(int)
result.to_csv("prediction_savvy_sea_lion_3.csv", index=False)
