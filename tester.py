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
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV

# Set a fixed random seed to make the work reproducible
np.random.seed(2022)


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

# df = df[df["CUSTOMER"].notna()]  # using this line will make the result set wrong
df["ISIC"] = df["ISIC"].ffill().bfill()

# Dealing with the dates
df["CREATION_YEAR"] = pd.to_datetime(df["CREATION_YEAR"]).dt.year
# TODO: DATE MATTER
df["MO_CREATED_YEAR"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.year
df["MO_CREATED_MONTH"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.month
df["SO_CREATED_YEAR"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.year
df["SO_CREATED_MONTH"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.month


df["CREATION_YEAR"] = df["CREATION_YEAR"].ffill().bfill()

# Uniting currency with cny
df["CURRENCY"] = df["CURRENCY"].map({"Chinese Yuan": 1, "Euro": 7.25, "US Dollar": 6.36, "Pound Sterling": 8.72, 0: 0})
for col in ["CURRENCY", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2"]:
    df[col] = df[col].fillna(df[col].mean())

df["REV_CURRENT_YEAR"] = df["REV_CURRENT_YEAR"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.1"] = df["REV_CURRENT_YEAR.1"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.2"] = df["REV_CURRENT_YEAR.2"] * df["CURRENCY"]

# Better to use unknown
df["COUNTRY"] = df["COUNTRY"].fillna("UNKNOWN")
df["OWNERSHIP"] = df["OWNERSHIP"].fillna("UNKNOWN")

df["END_CUSTOMER"] = df["END_CUSTOMER"].map(map_end_customer)

# Turning high/low margin products to percentage and make them comparable(effective)
for letter in ["A", "B", "C", "D", "E"]:
    df["Percentage_of_Product_" + letter] = df["COSTS_PRODUCT_" + letter] / (df["OFFER_PRICE"])
    df = df.drop(columns=["COSTS_PRODUCT_" + letter])

# One-hot encoding for nominal
df = pd.get_dummies(df, columns=["TECH", "BUSINESS_TYPE",
                                 "SALES_BRANCH", "PRICE_LIST", "SALES_OFFICE", "COUNTRY", "OWNERSHIP", "OFFER_TYPE",
                                 "SALES_LOCATION", "CURRENCY"])

# Drop useless variables
df = df.drop(columns=["MO_ID", "SO_ID", "CUSTOMER", "TEST_SET_ID"])

# Modeling the data

# dividing the outcomes and variables
df = df[df["OFFER_STATUS"].notna()]
Y = df["OFFER_STATUS"]
X = df.drop(columns="OFFER_STATUS")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# the scaling matter, for good habits
sc = MinMaxScaler(feature_range=(0, 100))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training and fit the best tree in the model
rf = RandomForestClassifier(n_estimators=200, class_weight={1: 1, 0: 4}, min_samples_split=0.001,
                            min_samples_leaf=0.0005)
logit = LogisticRegression(max_iter=2000, class_weight={1: 1, 0: 4}, solver="saga", C=0.01)
et = ExtraTreesClassifier(n_estimators=200, class_weight={1: 1, 0: 4}, min_samples_split=0.001,
                          min_samples_leaf=0.0005)
nb = ComplementNB(alpha=0.1)
hist = HistGradientBoostingClassifier()
grad = GradientBoostingClassifier()
voting = VotingClassifier(estimators=[('rf', rf), ('et', et)], voting='hard')

classifier = rf
classifier.fit(X_train, Y_train)

# cross validation
# cv_results = cross_validate(classifier, X, Y, cv=10, return_train_score=True)
# print(cv_results['test_score'])

Y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, \
    balanced_accuracy_score, accuracy_score

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# GridSearch for paras


balance = balanced_accuracy_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
specificity = 2 * balance - recall

print('accuracy: ' + str(accuracy_score(Y_test, Y_pred)))
print('balanced accuracy: ' + str(balance))
print('recall: ' + str(recall))
print('specificity: ' + str(specificity))

print(df['OFFER_STATUS'].count())
print(df['OFFER_STATUS'].sum())

"""classifiert = RandomForestClassifier(class_weight={1: 1, 0: 4})
paras = {"n_estimators": [10, 20, 50, 100, 200], "min_samples_split": [0.01, 0.05, 0.1, 0.001],
         "max_depth": [2, 4, None, 8, 16]}
cv = GridSearchCV(classifiert, paras, cv=5)
cv.fit(X_train, Y_train)
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


display(cv)
"""
