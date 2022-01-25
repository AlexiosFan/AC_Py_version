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
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


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

def map_isic(entry):
    if entry in range(1, 4):
        return 'A'
    elif entry in range(5, 10):
        return 'B'
    elif entry in range(10, 34):
        return 'C'
    elif entry == 35:
        return 'D'
    elif entry in range(36, 40):
        return 'E'
    elif entry in range(41, 44):
        return 'F'
    elif entry in range(45, 48):
        return 'G'
    elif entry in range(49, 54):
        return 'H'
    elif entry in range(55, 57):
        return 'I'
    elif entry in range(58, 64):
        return 'J'
    elif entry in range(64, 67):
        return 'K'
    elif entry == 68:
        return 'L'
    elif entry in range(69, 76):
        return 'M'
    elif entry in range(77, 83):
        return 'N'
    elif entry == 84:
        return 'O'
    elif entry == 85:
        return 'P'
    elif entry in range(85, 89):
        return 'Q'
    elif entry in range(90, 94):
        return 'R'
    elif entry in range(94, 97):
        return 'S'
    elif entry in range(97, 99):
        return 'T'
    elif entry == 99:
        return 'U'
    else:
        return np.nan


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

df["MO_CREATED_YEAR"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.year
df["MO_CREATED_MONTH"] = pd.to_datetime(df["MO_CREATED_DATE"]).dt.month

df["SO_CREATED_YEAR"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.year
df["SO_CREATED_MONTH"] = pd.to_datetime(df["SO_CREATED_DATE"]).dt.month


# Uniting currency with cny
df["CURRENCY"] = df["CURRENCY"].map({"Chinese Yuan": 1, "Euro": 7.2, "US Dollar": 6.4, "Pound Sterling": 8.6, 0: 0})

knn = KNNImputer()
revs = knn.fit_transform(df.get(["CURRENCY", "REV_CURRENT_YEAR", "REV_CURRENT_YEAR.1", "REV_CURRENT_YEAR.2", "CREATION_YEAR"]))
revs = pd.DataFrame(revs)
df["CURRENCY"] = revs[0]
df["REV_CURRENT_YEAR"] = revs[1]
df["REV_CURRENT_YEAR.1"] = revs[2]
df["REV_CURRENT_YEAR.2"] = revs[3]
df["CREATION_YEAR"] = revs[4]

df["REV_CURRENT_YEAR"] = df["REV_CURRENT_YEAR"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.1"] = df["REV_CURRENT_YEAR.1"] * df["CURRENCY"]
df["REV_CURRENT_YEAR.2"] = df["REV_CURRENT_YEAR.2"] * df["CURRENCY"]

#df["REVENUE"] = (df["OFFER_PRICE"]-df["SERVICE_LIST_PRICE"]-df["MATERIAL_COST"]-df["SERVICE_COST"])/df["OFFER_PRICE"]
#df["BARGAIN"] = (df["OFFER_PRICE"]-df["SERVICE_LIST_PRICE"])/df["OFFER_PRICE"]

df["END_CUSTOMER"] = df["END_CUSTOMER"].map(map_end_customer)

for entry in ['A', 'B', 'C', 'D', 'E']:
    df["Percentage_cost_"+entry] = df["COSTS_PRODUCT_"+entry]/df["MATERIAL_COST"]
    df["Percentage_cost_"+entry] = df["Percentage_cost_"+entry].fillna(0).replace(np.inf, 0)
    df = df.drop(columns=["COSTS_PRODUCT_"+entry])

df["ISIC"] = (df["ISIC"].fillna(0)/100).map(map_isic)

imputer = SimpleImputer(strategy="most_frequent")
subres = imputer.fit_transform(df.get(["COUNTRY", "OWNERSHIP", "ISIC"]))
subres = pd.DataFrame(subres)
df["COUNTRY"] = subres[0]
df["OWNERSHIP"] = subres[1]


# One-hot encoding is bad in random forest, alternatively label encoding is better

label_encoder = LabelEncoder()
for feature in ["TECH", "BUSINESS_TYPE", "PRICE_LIST", "OWNERSHIP", "OFFER_TYPE", "SALES_BRANCH",
                "SALES_LOCATION", "CURRENCY", "ISIC", "COUNTRY"]:
    df[feature] = label_encoder.fit_transform(df[feature])

# Drop useless variables
df = df.drop(columns=["MO_ID", "SO_ID", "CUSTOMER", "TEST_SET_ID", "MO_CREATED_DATE", "SO_CREATED_DATE", "END_CUSTOMER",
                      "SALES_OFFICE"])



# Modeling the data

# dividing the outcomes and variables
df = df[df["OFFER_STATUS"].notna()]
Y = df["OFFER_STATUS"]
X = df.drop(columns="OFFER_STATUS")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=random_state)

# the scaling matter, for good habits
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
# training and fit the best tree in the model
rf = BalancedRandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", sampling_strategy="all",
                                    criterion='entropy', random_state=random_state)
logit = LogisticRegression(max_iter=2000, class_weight={1: 1, 0: 4.5}, solver="saga", C=0.01)
et = ExtraTreesClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=random_state)

voting = VotingClassifier(estimators=[('rf', rf), ('et', et)], voting="hard")

classifier = rf
classifier.fit(X_train, Y_train)

# cross validation
#cv_results = cross_validate(classifier, X, Y, cv=3, scoring="balanced_accuracy")
#print(cv_results['test_score'])

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

"""
classifiert = RandomForestClassifier(class_weight={1: 1, 0: 4})
paras = {"n_estimators": [100, 200], "min_samples_split": [0.002, 0.001, 0.004],
         "min_samples_leaf": [0.001, 0.0005, 0.002, 0.001]}
cv = GridSearchCV(classifiert, paras, cv=3)
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
