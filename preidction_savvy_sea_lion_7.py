"""
the classification/regression part is partially quoted from the source
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
with regards and thanks to the tutorial of usage of sklearning package posts
"""
# Loading packages and data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, \
    StackingClassifier, HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier

random_state = 2022
# Set a fixed random seed to make the work reproducible
np.random.seed(random_state)


# mapping functions
# Converting string to nums
def remove_quote(entry):
    return entry[1:-1]


# removing nv from the datasets
def remove_NV(entry):
    if entry in ['#NV', 'NA', np.nan]:
        return np.nan
    else:
        return float(entry)


#
# Process the end_customers
def map_end_customer(entry):
    if entry in ["No"]:
        return 0
    elif entry in ["Yes", np.nan]:
        return 1
    else:
        return 0


# read files
transactions = pd.read_csv("transactions.csv")
customers = pd.read_csv("customers.csv")
geo = pd.read_csv("geo.csv")

# Join transactions with geo, customers
# We use the left outer join to ensure there is no data loss in transactions
df = transactions.merge(geo, how="left")

# The data type of attribute "CUSTOMER" in data transactions is String
# We need to firstly convert the data type into int
df["CUSTOMER"] = df["CUSTOMER"].map(remove_quote, 'ignore').map(remove_NV).astype(int, errors="ignore")

# The data type of "REV_CURRENT_YEAR" is string
# We need to convert the data type into float
customers["REV_CURRENT_YEAR"] = customers["REV_CURRENT_YEAR"].map(remove_quote).astype(float, errors="ignore")

# The format of attribute "COUNTRY" is different in data customers and transactions
# We need to firstly convert them to the same format
df["COUNTRY"] = df["COUNTRY"].map({"CH": "Switzerland", "FR": "France"})
df = df.merge(customers, how="left", on=["CUSTOMER", "COUNTRY"])

# Exploratory Analysis and Preprocessing

# data preparation for OFFER_STATUS
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["LOST", "Lost", "LOsT", "Lose"], 0.)
df["OFFER_STATUS"] = df["OFFER_STATUS"].replace(["WIN", "Win", "Won", "WON"], 1.)

# Dealing with the dates
df["CREATION_YEAR"] = pd.to_datetime(df["CREATION_YEAR"]).dt.year
df["CREATION_YEAR"] = df["CREATION_YEAR"].fillna(df["CREATION_YEAR"].mean())

df["MO_CREATED_YEAR"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.year
df["MO_CREATED_MONTH"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.month

df["SO_CREATED_YEAR"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.year
df["SO_CREATED_MONTH"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.month


# Uniting currency with cny
df["CURRENCY"] = df["CURRENCY"].map({"Chinese Yuan": 1, "Euro": 7.2, "US Dollar": 6.4, "Pound Sterling": 8.6, 0: 0})
for col in ["CURRENCY", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2"]:
    df[col] = df[col].fillna(df[col].mean())

df["REV_CURRENT_YEAR"] = df["REV_CURRENT_YEAR"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.1"] = df["REV_CURRENT_YEAR.1"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.2"] = df["REV_CURRENT_YEAR.2"] * df["CURRENCY"]

df["REVENUE"] = (df["OFFER_PRICE"]-df["SERVICE_LIST_PRICE"]-df["MATERIAL_COST"]-df["SERVICE_COST"])/df["OFFER_PRICE"]


# Better to use unknown
df["COUNTRY"] = df["COUNTRY"].fillna("UNKNOWN")
df["OWNERSHIP"] = df["OWNERSHIP"].fillna("UNKNOWN")

df["END_CUSTOMER"] = df["END_CUSTOMER"].map(map_end_customer)
df["ISIC"] = (df["ISIC"].fillna(0)/100)

# One-hot encoding is bad in random forest, alternatively label encoding is better

label_encoder = LabelEncoder()
for feature in ["TECH", "BUSINESS_TYPE", "PRICE_LIST", "OWNERSHIP", "OFFER_TYPE", "SALES_BRANCH",
                "SALES_LOCATION", "CURRENCY"]:
    df[feature] = label_encoder.fit_transform(df[feature])

# Drop useless variables
df = df.drop(columns=["MO_ID", "SO_ID", "CUSTOMER", "MO_CREATED_DATE", "SO_CREATED_DATE", "END_CUSTOMER",
                      "COUNTRY", "SALES_OFFICE"])

# Modeling the data

# separate the training set and the test set
train_set = df[df["OFFER_STATUS"].notna()].drop(columns="TEST_SET_ID")
test_set_init = df[df["OFFER_STATUS"].isna()]  # used to generate the outcome

test_set = test_set_init.drop(columns="TEST_SET_ID")

# Dividing the variables and outcomes
Y_train = train_set["OFFER_STATUS"].values
X_train = train_set.drop(columns="OFFER_STATUS").values

X_test = test_set.drop(columns="OFFER_STATUS").values

# the scaling matter, for good habits
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training and fit the best tree in the model
rf = BalancedRandomForestClassifier(n_estimators=128, class_weight="balanced_subsample", sampling_strategy="all",
                                    criterion='entropy')

rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

# Output the results
result = pd.DataFrame([test_set_init["TEST_SET_ID"].to_numpy(), np.split(Y_pred, len(Y_pred))],
                      index=["id", "prediction"]).T
result["prediction"] = result["prediction"].map(np.sum)
result = result.astype(int)
result.to_csv("prediction_savvy_sea_lion_7.csv", index=False)




