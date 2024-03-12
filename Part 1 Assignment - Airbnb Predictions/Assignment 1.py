#
# Getting the data and performing some initial analysis part 1
#

# Importing pandas for plots and graphs and numpy for scientific computing with Python
import pandas as pd
import numpy as np

# Having copied file and stored it locally as a csv file in the working directory
# Reading the airbnb excel file
airbnb = pd.read_csv("Part 1 Assignment - Airbnb Predictions\\BristolAirbnbListings.csv")

# Get some details about it the airbnb data file
# First few lines
airbnb.head()

# Getting information on the data set as a whole
airbnb.info()

# Predicting price, so I have chosen to look at the value counts for that
# column alone.
airbnb["price"].value_counts()

# Summary stats for the data airbnb
airbnb.describe()


# Importing matplot package for graphs, histograms and any other plots that 
# that I may or may not want to use.
import matplotlib.pyplot as plt

# Histogram for the data set and its labels to see what factors and value
# ratings are involved or connected to the data.
# when running this using Jupyter notebook then also include the line %matplotlib inline
airbnb.hist(bins=50, figsize=(12,7))
%matplotlib inline
plt.show()








#
# Preparing the data such as cleaning, getting rid of nulls and objects, 
# changing the nulls to averages and the objects to nulls which are then changed
# to averages. This sections also drops columns, gets the values for each attribute
# and changes the results to float numerical value from objects for later on
# so they are still not recognized as objects.

# checking for nulls in the data set.
airbnb.isnull().any()

# Checking the data for value counts, each column is selected to get a good idea
# of what each part of the data has.
airbnb["id"].value_counts()
airbnb["name"].value_counts()
airbnb["host_id"].value_counts()
airbnb["host_name"].value_counts()
airbnb["neighbourhood"].value_counts()
airbnb["postcode"].value_counts()
airbnb["latitude"].value_counts()
airbnb["longitude"].value_counts()
airbnb["property_type"].value_counts()
airbnb["room_type"].value_counts()
airbnb["accommodates"].value_counts()
airbnb["bathrooms"].value_counts()
airbnb["bedrooms"].value_counts()
airbnb["beds"].value_counts()
airbnb["price"].value_counts()
airbnb["minimum_nights"].value_counts()
airbnb["number_of_reviews"].value_counts()
airbnb["last_review"].value_counts()
airbnb["reviews_per_month"].value_counts()
airbnb["review_scores_rating"].value_counts()
airbnb["review_scores_accuracy"].value_counts()
airbnb["review_scores_cleanliness"].value_counts()
airbnb["review_scores_checkin"].value_counts()
airbnb["review_scores_communication"].value_counts()
airbnb["review_scores_location"].value_counts()
airbnb["review_scores_value"].value_counts()
airbnb["calculated_host_listings_count"].value_counts()
airbnb["availability_365"].value_counts()


# Dropping the columns which are not commented out, I have decided to base the 
# search for a predictions on price with mostly reviews along with locations and
# size of the bedrooms, accommocations and other factors commented out below. 
airbnb = airbnb.drop("id", axis=1)       # option 2
airbnb = airbnb.drop("name", axis=1)
airbnb = airbnb.drop("host_id", axis=1)
airbnb = airbnb.drop("host_name", axis=1)
airbnb = airbnb.drop("neighbourhood", axis=1)
airbnb = airbnb.drop("postcode", axis=1)
#airbnb = airbnb.drop("latitude", axis=1)
#airbnb = airbnb.drop("longitude", axis=1)
airbnb = airbnb.drop("property_type", axis=1)
airbnb = airbnb.drop("room_type", axis=1)
#airbnb = airbnb.drop("accommodates", axis=1)
#airbnb = airbnb.drop("bathrooms", axis=1)
#airbnb = airbnb.drop("bedrooms", axis=1)
#airbnb = airbnb.drop("beds", axis=1)
#airbnb = airbnb.drop("price", axis=1)
#airbnb = airbnb.drop("minimum_nights", axis=1)
#airbnb = airbnb.drop("number_of_reviews", axis=1)
airbnb = airbnb.drop("last_review", axis=1)
#airbnb = airbnb.drop("reviews_per_month", axis=1)
#airbnb = airbnb.drop("review_scores_rating", axis=1)
#airbnb = airbnb.drop("review_scores_accuracy", axis=1)
#airbnb = airbnb.drop("review_scores_cleanliness", axis=1)
#airbnb = airbnb.drop("review_scores_checkin", axis=1)
#airbnb = airbnb.drop("review_scores_communication", axis=1)
airbnb = airbnb.drop("review_scores_location", axis=1)
airbnb = airbnb.drop("review_scores_value", axis=1)
airbnb = airbnb.drop("calculated_host_listings_count", axis=1)
airbnb = airbnb.drop("availability_365", axis=1)

# Always check the information of the data set before moving on to see if the 
# code has worked.
airbnb.info()


# Ways of dealing with missing values...

# Changing objects to null numbers, and then all null numbers changed to averages,
# so there is no missing data or entries that will slow down or stop the methods
# from working. Ive ran a for loop for each attribute that doesnt have 2375 values
# that dont include null, changing them to averages.

# accommodates
row = 0
for i in airbnb["accommodates"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"accommodates"]=np.nan
    row+=1
    
median = airbnb["accommodates"].median()
airbnb["accommodates"].fillna(median, inplace=True) # option 3
airbnb

# bathrooms
row = 0
for i in airbnb["bathrooms"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"bathrooms"]=np.nan
    row+=1
    
median = airbnb["bathrooms"].median()
airbnb["bathrooms"].fillna(median, inplace=True) # option 3
airbnb

# bedrooms
row = 0
for i in airbnb["bedrooms"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"bedrooms"]=np.nan
    row+=1
    
median = airbnb["bedrooms"].median()
airbnb["bedrooms"].fillna(median, inplace=True) # option 3
airbnb

# beds
row = 0
for i in airbnb["beds"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"beds"]=np.nan
    row+=1
    
median = airbnb["beds"].median()
airbnb["beds"].fillna(median, inplace=True) # option 3
airbnb

# reviews_per_month
row = 0
for i in airbnb["reviews_per_month"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"reviews_per_month"]=np.nan
    row+=1
    
median = airbnb["reviews_per_month"].median()
airbnb["reviews_per_month"].fillna(median, inplace=True) # option 3
airbnb

# review_scores_rating
row = 0
for i in airbnb["review_scores_rating"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"review_scores_rating"]=np.nan
    row+=1
    
median = airbnb["review_scores_rating"].median()
airbnb["review_scores_rating"].fillna(median, inplace=True) # option 3
airbnb

# review_scores_accuracy
row = 0
for i in airbnb["review_scores_accuracy"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"review_scores_accuracy"]=np.nan
    row+=1
    
median = airbnb["review_scores_accuracy"].median()
airbnb["review_scores_accuracy"].fillna(median, inplace=True) # option 3
airbnb

# review_scores_cleanliness
row = 0
for i in airbnb["review_scores_cleanliness"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"review_scores_cleanliness"]=np.nan
    row+=1
    
median = airbnb["review_scores_cleanliness"].median()
airbnb["review_scores_cleanliness"].fillna(median, inplace=True) # option 3
airbnb

# review_scores_checkin
row = 0
for i in airbnb["review_scores_checkin"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"review_scores_checkin"]=np.nan
    row+=1
    
median = airbnb["review_scores_checkin"].median()
airbnb["review_scores_checkin"].fillna(median, inplace=True) # option 3
airbnb

# review_scores_communication
row = 0
for i in airbnb["review_scores_communication"]:
    try:
        float(i)
    except ValueError:
        airbnb.loc[row,"review_scores_communication"]=np.nan
    row+=1
    
median = airbnb["review_scores_communication"].median()
airbnb["review_scores_communication"].fillna(median, inplace=True) # option 3
airbnb

# Looking at the information of the data set again to see what the code has 
# and if it has worked before moving on.
airbnb.info()





# Further later in the methods, the attributes may still be recognised as objects,
# even though there are no null values or objects in the rows, the data still has
# to be recognized as being of float or int numerical value.
airbnb["accommodates"] = airbnb["accommodates"].astype(np.float64)
airbnb["bathrooms"] = airbnb["bathrooms"].astype(np.float64)
airbnb["bedrooms"] = airbnb["bedrooms"].astype(np.float64)
airbnb["beds"] = airbnb["beds"].astype(np.float64)
airbnb["review_scores_rating"] = airbnb["review_scores_rating"].astype(np.float64)
airbnb["review_scores_accuracy"] = airbnb["review_scores_accuracy"].astype(np.float64)
airbnb["review_scores_cleanliness"] = airbnb["review_scores_cleanliness"].astype(np.float64)
airbnb["review_scores_checkin"] = airbnb["review_scores_checkin"].astype(np.float64)

# Again... checking the data is correct before moving o
airbnb.info()









#
# Investigate correlations to see relationships between price and other attributes
#

# Checking the correlations betwene price and all the attributes left in the data
# after the rest being dropped. This will help get a better view/representation
# of which attributes are related to price.
corr_matrix = airbnb.corr()
corr_matrix
corr_matrix["price"].sort_values(ascending=False)

# plotting correlations between all the data using pandas plotting and scatter matrix
from pandas.plotting import scatter_matrix

# The first half of the attributes, since using them all together would mean
# the graph is too big to view or judge.
attributes = ["accommodates", "beds", "bedrooms", "bathrooms", 
              "review_scores_cleanliness", "review_scores_checkin", 
              "review_scores_rating"]
scatter_matrix(airbnb[attributes], figsize=(12,7))
plt.show()

# Second half of the attributes...
attributes = ["latitude", "review_scores_communication",
              "minimum_nights", "review_scores_accuracy", "reviews_per_month",
              "number_of_reviews", "longitude"]
scatter_matrix(airbnb[attributes], figsize=(12,7))
plt.show()







#
# Visualising the data based on price
#

# scatter pots for the attributes chosen to see their relationship with price
# with alpha level 0.1
airbnb.plot(kind="scatter", x="price", y="accommodates", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="beds", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="bedrooms", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="bathrooms", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="review_scores_cleanliness", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="review_scores_checkin", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="review_scores_rating", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="latitude", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="review_scores_communication", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="minimum_nights", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="review_scores_accuracy", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="reviews_per_month", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="number_of_reviews", alpha=0.1)
airbnb.plot(kind="scatter", x="price", y="longitude", alpha=0.1)
plt.show()










#
# Create training and test sets using the Stratified Shuffle Split. Splitting
# up the data into an 80% training and the remaining 20% for testing. The shuffle
# creates a new column called price_cat where the data is then mad
#

# Looking at spread of price
airbnb["price"].hist()
plt.show()

# Introduce a new column in the data frame...
# Divide by 1.5 to limit the number of income categories
airbnb["price_cat"] = np.ceil(airbnb["price"] / 50)
# Label those above 5 as 5
airbnb["price_cat"].where(airbnb["price_cat"] < 5, 5.0, inplace=True)
airbnb["price_cat"].where(airbnb["price_cat"] > 1, 1.0, inplace=True)

airbnb["price_cat"].hist()
plt.show()

# Use StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# split generated two sets of indices based on preserving distribution of price_cat attribute 
# use these to create train and test sets
for train_index, test_index in split.split(airbnb, airbnb["price_cat"]):
    strat_train_set = airbnb.loc[train_index]
    strat_test_set = airbnb.loc[test_index]

strat_train_set.describe()
strat_test_set.describe()

# after dropping labels from the training set and create a separate data frame with the target variable
airbnb_labels = strat_train_set["price"].copy()

# check histograms...
strat_train_set.hist(bins=50, figsize=(12,7))
strat_test_set.hist(bins=50, figsize=(12,7))
plt.show()

# Having created these can then remove the income_cat column
# One way
for set_ in (strat_train_set, strat_test_set):
    set_.drop("price_cat", axis=1, inplace=True)

# check histograms...again after dropping price_cat
strat_train_set.hist(bins=50, figsize=(12,7))
strat_test_set.hist(bins=50, figsize=(12,7))
plt.show()

# Or more simply.. (axis designates column)
#strat_train_set.drop("income_cat", axis=1, inplace=True)
#strat_test_set.drop("income_cat", axis=1, inplace=True)
    

# New Variable names for strat train set, dropping price and putting it into
# a new variables called airbnb train labels
airbnbTrain = strat_train_set
airbnbTrain.info()
airbnbTrainLabels = airbnbTrain["price"]
airbnbTrain = airbnbTrain.drop("price", axis=1)
airbnbTrain.info()

airbnbTrainLabels



# New Variable names for strat test set, dropping price and putting it into
# a new variables called airbnb train labels
airbnbTest = strat_test_set
airbnbTest.info()
airbnbTestLabels = airbnbTest["price"]
airbnbTest = airbnbTest.drop("price", axis=1)
airbnbTest.info()

airbnbTestLabels


airbnbLabels = airbnb["price"]
airbnb = airbnb.drop("price", axis=1)










#
# My chosen models for calcuating predictions based on the data price.
#


# LINEAR REGRESSION

# Try out linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(airbnbTrain, airbnbTrainLabels)

# Looking at just a subset of the data for illustrative purposes
some_data = airbnbTrain.iloc[:5]
some_labels = airbnbTrainLabels.iloc[:5]

print("Predictions:", lin_reg.predict(some_data))

print("Labels:", list(some_labels))









# Support Vector Machine Regression
from sklearn.svm import SVR
# Create and fit the classifier to the scaled data

svm_reg = SVR(C=700, gamma = 5, epsilon = 0.1)
svm_reg.fit(airbnbTrain, airbnbTrainLabels)

some_data = airbnbTrain.iloc[:5]
some_labels = airbnbTrainLabels.iloc[:5]

print("Predictions:", svm_reg.predict(some_data))

print("Labels:", list(some_labels))








# Random Forrest 
from sklearn.ensemble import RandomForestRegressor
rnd_reg = RandomForestRegressor(n_estimators=500, n_jobs=-1)

rnd_reg.fit(airbnbTrain, airbnbTrainLabels)

some_data = airbnbTrain.iloc[:5]
some_labels = airbnbTrainLabels.iloc[:5]

print("Predictions:", rnd_reg.predict(some_data))

print("Labels:", list(some_labels))



# Bagging
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bag_reg = BaggingRegressor(
    DecisionTreeRegressor(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)

bag_reg.fit(airbnbTrain, airbnbTrainLabels)

some_data = airbnbTrain.iloc[:5]
some_labels = airbnbTrainLabels.iloc[:5]

print("Predictions:", bag_reg.predict(some_data))

print("Labels:", list(some_labels))




# Desicion tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=5)
tree_reg.fit(airbnbTrain, airbnbTrainLabels)

some_data = airbnbTrain.iloc[:5]
some_labels = airbnbTrainLabels.iloc[:5]

print("Predictions:", tree_reg.predict(some_data))

print("Labels:", list(some_labels))

# visualizing decision tree decision tree
import graphviz

export_graphviz(rf_regTree,feature_names = num_attribs, out_file='rf_RegTree.dot',
                rounded=True,proportion=False, precision=2, filled=True)

call(['dot', '-Tpng', 'rf_RegTree.dot', '-o', 'rf_RegTree.png', '-Gdpi=600'])















#
# Calculating RMSE for all the methods and printing them 
#
from sklearn.metrics import mean_squared_error

# Linear regression
airbnbTrainPredictions = lin_reg.predict(airbnbTrain)

lin_mse = mean_squared_error(airbnbTrainLabels, airbnbTrainPredictions)
lin_rmse = np.sqrt(lin_mse)
print("root means square error is:", lin_rmse)

# Support Vector Machine regression
airbnbsupportPredictions = svm_reg.predict(airbnbTrain)

svm_mse = mean_squared_error(airbnbTrainLabels, airbnbsupportPredictions)
svm_rmse = np.sqrt(svm_mse)
print("root means square error is:", svm_rmse)


# Random Forrest
airbnbrandomPredictions = rnd_reg.predict(airbnbTrain)

rnd_mse = mean_squared_error(airbnbTrainLabels, airbnbrandomPredictions)
rnd_rmse = np.sqrt(rnd_mse)
print("root means square error is:", rnd_rmse)

# Bagging
airbnbbaggingPredictions = bag_reg.predict(airbnbTrain)

bag_mse = mean_squared_error(airbnbTrainLabels, airbnbbaggingPredictions)
bag_rmse = np.sqrt(bag_mse)
print("root means square error is:", bag_rmse)



# Decision
airbnbdecisionPredictions = tree_reg.predict(airbnbTrain)

tree_mse = mean_squared_error(airbnbTrainLabels, airbnbdecisionPredictions)
tree_rmse = np.sqrt(tree_mse)
print("root means square error is:", tree_rmse)







# Chosen model and best RMSE for test
# Decision
airbnbtesttreePredictions = tree_reg.predict(airbnbTest)

tree_test_mse = mean_squared_error(airbnbTestLabels, airbnbtesttreePredictions)
tree_test_rmse = np.sqrt(tree_test_mse)
print("root means square error is:", tree_test_rmse)

# Bagging
airbnbtestbagPredictions = bag_reg.predict(airbnbTest)

bag_test_mse = mean_squared_error(airbnbTestLabels, airbnbtestbagPredictions)
bag_test_mse = np.sqrt(bag_test_mse)
print("root means square error is:", bag_test_mse)









# this is for whole data set,holds all predictions for whole dataset
airbnbPredictions = bag_reg.predict(airbnb)

bag_test_mse = mean_squared_error(airbnbLabels, airbnbPredictions)
bag_test_mse = np.sqrt(bag_test_mse)
print("root means square error is:", bag_test_mse)





# this plots all predictions
plt.plot(airbnbPredictions)
plt.ylabel('some numbers')
plt.show()


# this plots all labels
plt.plot(airbnbLabels)
plt.ylabel('some numbers')
plt.show()




