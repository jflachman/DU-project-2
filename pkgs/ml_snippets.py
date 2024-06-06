

# ----------------------------------------------------------------
# ----- Dataframe examination
# ----------------------------------------------------------------
# Count values of 'y'
df['y'].value_counts()

#Verify changes with info
df_clean.info()

# Find the percentage of rows missing values in each column
X_train.isna().sum()/len(df)

# Describe the other columns in the rows with missing values
X_train.loc[X_train['backers_count'].isna()].describe()

# Describe the whole dataset
X_train.describe()

# Perform other exploratory analysis here
# For this specific example, try making histograms
# of days_active for the whole dataset and then
# again only when backers_count is missing.

X_train['days_active'].plot(kind='hist', alpha=0.2)
X_train.loc[df['backers_count'].isna(), 'days_active'].plot(kind='hist')
print(X_train.loc[df['backers_count'].isna(), 'days_active'].unique())

# Check for missing values
print(X_train_clean.isna().sum()/len(X_train_clean))
print(X_test_clean.isna().sum()/len(X_test_clean))

# Find the percentage of null values in each column
X_train.isna().sum()/len(X_train)

# Next is pdays
# This column says how many days it has been since the last 
# marketing contact for this client
X_train['pdays'].plot(kind='hist')

# describe null data in 'days'
X_train.loc[X_train['pdays'].isna()].describe()



# ----------------------------------------------------------------
# ----- Encoding / Filling
# ----------------------------------------------------------------
# Convert y to numeric
df_clean['y'] = pd.get_dummies(df['y'], drop_first=True, dtype='int')

#Drop all non-numeric columns
df_clean = df_clean.select_dtypes(include='number')

# Since backers_count seems to be missing in the first week
# of a campaign, removing the data would be detrimental.
# A good choice might be to fill the data using the backers
# counts from campaigns in week 2.

mean_of_week_2_backers_counts = X_train.loc[(X_train['days_active'] >= 6) & (X_train['days_active'] <= 13), 'backers_count'].mean()
mean_of_week_2_backers_counts

# Create a function to fill missing values with half of the mean of week 2
def X_preprocess(X_data):
    X_data['backers_count'] = X_data['backers_count'].fillna(int(round(mean_of_week_2_backers_counts/2)))
    return X_data

# Create an encoder for the backpack_color column
backpack_color_ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
# Train the encoder
backpack_color_ohe.fit(X_train['backpack_color'].values.reshape(-1,1))



# Add single column encoders
X_data_encoded['education'] = encode_education.transform(X_data['education'].values.reshape(-1, 1))


# Fill the missing values with zeros
df['pledged_per_backer'] = df['pledged_per_backer'].fillna(0)

# Create a days_to_goal column
def days_to_goal(row):
    amount_remaining = row['goal'] - row['pledged']
    pledged_per_day = row['pledged_per_backer'] * row['backers_per_day']
    # Note that we can't divide by zero:
    # return a large number if pledged_per_day is zero
    if pledged_per_day == 0:
        return 10000
    return (amount_remaining)/(pledged_per_day)

df['days_to_goal'] = df.apply(days_to_goal, axis=1)

# The job column is varied and the number of missing values is small
# It might suffice to fill the missing values with "unknown"
# We'll make a function to handle this.
def fill_job(X_data):
    X_data['job'] = X_data['job'].fillna('unknown')
    return X_data

# ----------------------------
#----- Encoding example 
# Create a OneHotEncoder
encode_y = OneHotEncoder(drop='first', sparse_output=False)
# Train the encoder
encode_y.fit(y_train)
# Apply it to both y_train and y_test
# Use np.ravel to reshape for logistic regression
y_train_encoded = np.ravel(encode_y.transform(y_train))
y_test_encoded = np.ravel(encode_y.transform(y_test))
y_train_encoded


# ----------------------------
#----- Encoding example 
# Lets make this an Ordinal column
encode_housing= OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_housing.fit(X_train_filled['housing'].values.reshape(-1, 1))
# This is ordinal! Lets use the ordinal encoder
# We'll set any unknown values to -1
encode_education = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_education.fit(X_train_filled['education'].values.reshape(-1, 1))
# This month seems ordinal by may not behave that way...
# Lets use ordinal for now, but consider experimenting with this!
encode_month = OrdinalEncoder(categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_month.fit(X_train_filled['month'].values.reshape(-1, 1))

# Combine the encoders into a function
# Make sure to return a dataframe
def encode_categorical(X_data):
    # Separate numeric columns
    X_data_numeric = X_data.select_dtypes(include='number').reset_index()

    # Multicolumn encoders first
    job_encoded_df = pd.DataFrame(encode_job.transform(X_data['job'].values.reshape(-1, 1)), columns=encode_job.get_feature_names_out())
    marital_encoded_df = pd.DataFrame(encode_marital.transform(X_data['marital'].values.reshape(-1, 1)), columns=encode_marital.get_feature_names_out())
    contact_encoded_df = pd.DataFrame(encode_contact.transform(X_data['contact'].values.reshape(-1, 1)), columns=encode_contact.get_feature_names_out())
    poutcome_encoded_df = pd.DataFrame(encode_poutcome.transform(X_data['poutcome'].values.reshape(-1, 1)), columns=encode_poutcome.get_feature_names_out())

    # Concat all dfs together
    dfs = [X_data_numeric, job_encoded_df, marital_encoded_df, contact_encoded_df, poutcome_encoded_df]
    X_data_encoded = pd.concat(dfs, axis=1)

    # Add single column encoders
    X_data_encoded['education'] = encode_education.transform(X_data['education'].values.reshape(-1, 1))
    X_data_encoded['default'] = encode_default.transform(X_data['default'].values.reshape(-1, 1))
    X_data_encoded['housing'] = encode_housing.transform(X_data['housing'].values.reshape(-1, 1))
    X_data_encoded['loan'] = encode_loan.transform(X_data['loan'].values.reshape(-1, 1))
    X_data_encoded['month'] = encode_month.transform(X_data['month'].values.reshape(-1, 1))

# ----------------------------
#----- Encoding example 
# Lots of unique values, not ordinal data
# Lets convert to no more than 5 categories
encode_job = OneHotEncoder(max_categories=5, handle_unknown='infrequent_if_exist', sparse_output=False)
# Train the encoder
encode_job.fit(X_train_filled['job'].values.reshape(-1, 1))
# Only three values; lets use two OneHotEncoded columns
# remembering to choose options for unknown values
encode_marital = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
encode_marital.fit(X_train_filled['marital'].values.reshape(-1, 1))
# This is ordinal! Lets use the ordinal encoder
# We'll set any unknown values to -1
encode_education = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_education.fit(X_train_filled['education'].values.reshape(-1, 1))
# Lets make this an Ordinal column
encode_default = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_default.fit(X_train_filled['default'].values.reshape(-1, 1))
# Lets make this an Ordinal column
encode_housing= OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_housing.fit(X_train_filled['housing'].values.reshape(-1, 1))
# Lets make this an Ordinal column
encode_loan = OrdinalEncoder(categories=[['no', 'yes']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_loan.fit(X_train_filled['loan'].values.reshape(-1, 1))
# Lets use two OneHotEncoded columns
encode_contact = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
encode_contact.fit(X_train_filled['contact'].values.reshape(-1, 1))
# This month seems ordinal by may not behave that way...
# Lets use ordinal for now, but consider experimenting with this!
encode_month = OrdinalEncoder(categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']], handle_unknown='use_encoded_value', unknown_value=-1)
encode_month.fit(X_train_filled['month'].values.reshape(-1, 1))
# Lets use OneHotEncoding for this
encode_poutcome = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
encode_poutcome.fit(X_train_filled['poutcome'].values.reshape(-1, 1))

# Combine the encoders into a function
# Make sure to return a dataframe
def encode_categorical(X_data):
    # Separate numeric columns
    X_data_numeric = X_data.select_dtypes(include='number').reset_index()

    # Multicolumn encoders first
    job_encoded_df = pd.DataFrame(encode_job.transform(X_data['job'].values.reshape(-1, 1)), columns=encode_job.get_feature_names_out())
    marital_encoded_df = pd.DataFrame(encode_marital.transform(X_data['marital'].values.reshape(-1, 1)), columns=encode_marital.get_feature_names_out())
    contact_encoded_df = pd.DataFrame(encode_contact.transform(X_data['contact'].values.reshape(-1, 1)), columns=encode_contact.get_feature_names_out())
    poutcome_encoded_df = pd.DataFrame(encode_poutcome.transform(X_data['poutcome'].values.reshape(-1, 1)), columns=encode_poutcome.get_feature_names_out())

    # Concat all dfs together
    dfs = [X_data_numeric, job_encoded_df, marital_encoded_df, contact_encoded_df, poutcome_encoded_df]
    X_data_encoded = pd.concat(dfs, axis=1)

    # Add single column encoders
    X_data_encoded['education'] = encode_education.transform(X_data['education'].values.reshape(-1, 1))
    X_data_encoded['default'] = encode_default.transform(X_data['default'].values.reshape(-1, 1))
    X_data_encoded['housing'] = encode_housing.transform(X_data['housing'].values.reshape(-1, 1))
    X_data_encoded['loan'] = encode_loan.transform(X_data['loan'].values.reshape(-1, 1))
    X_data_encoded['month'] = encode_month.transform(X_data['month'].values.reshape(-1, 1))
    
    return X_data_encoded

# Apply the encoding function to both training and testing
X_train_encoded = encode_categorical(X_train_filled)
X_test_encoded = encode_categorical(X_test_filled)

#------- Label endocder
# Get the target variable (the "Class" column)
# Since the target column is an object, we need to convert the data to numerical classes
# Use the LabelEncoder
# Create an instance of the label encoder
le = LabelEncoder()
y = le.fit_transform(df["Class"])



#------- 

#------- 

#------- 

#------- 


# ----------------------------------------------------------------
# ----- Setup X and Y variables
# ----------------------------------------------------------------
# Setup X and Y variables
X = df_clean.drop(columns=['target'])
y = df_clean['target']

# Split into training and testing sets
X = df.drop(columns = 'outcome')
y = df['outcome'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)

# ----------------------------------------------------------------
# ----- Models
# ----------------------------------------------------------------

# ----- 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,y)
model.score(X,y)

# -----
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
#model = RandomForestClassifier(random_state=13)
#model = RandomForestClassifier(n_estimators=500)
model.fit(X,y)
model.score(X,y)
classifier.score(X_test, y_test)
classifier.score(X_train, y_train)

# -----
# Create a Logistic Regression Model
classifier = LogisticRegression()
classifier.fit(X, y)
classifier.score(X, y)

# -----
# Create and score a decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
clf.score(iris.data, iris.target)

# -----
from sklearn.svm import SVC # Create the support vector machine classifier model with a 'linear' kernel
model = SVC(kernel='linear')
# Fit the model to the training data
model.fit(X_train, y_train)
# Validate the model by checking the model accuracy with model.score
print('Train Accuracy: %.3f' % model.score(X_train, y_train))
print('Test Accuracy: %.3f' % model.score(X_test, y_test))

# Display the accuracy score for the testing dataset
accuracy_score(y_test, testing_predictions)

# -----
from sklearn import tree
# Create the decision tree classifier instance
model = tree.DecisionTreeClassifier()
# Fit the model
model = model.fit(X_train_scaled, y_train)
# Making predictions using the testing data
predictions = model.predict(X_test_scaled)
# Calculate the accuracy score
acc_score = accuracy_score(y_test, predictions)
print(f"Accuracy Score : {acc_score}")

# --- Visualizing a decision tree
# Create DOT data
dot_data = tree.export_graphviz(
    model, out_file=None, feature_names=X.columns, class_names=["0", "1"], filled=True
)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
Image(graph.create_png())

# -----
# Train the Random Forest model - 13.2.5 *********
clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)
# Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')
# Get the feature importance array
feature_importances = clf.feature_importances_
# List the top 10 most important features
importances_sorted = sorted(zip(feature_importances, X.columns), reverse=True)
importances_sorted[:10]
# Plot the feature importances
features = sorted(zip(X.columns, feature_importances), key = lambda x: x[1])
cols = [f[0] for f in features]
width = [f[1] for f in features]
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
plt.margins(y=0.001)
ax.barh(y=cols, width=width)
plt.show()

# -----
# Import an Extremely Random Trees classifier
from sklearn.ensemble import ExtraTreesClassifier
# Train the ExtraTreesClassifier model
clf = ExtraTreesClassifier(random_state=1).fit(X_train_scaled, y_train)
# Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

#------
# Import Gradient Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
# Train the Gradient Boosting classifier
clf = GradientBoostingClassifier(random_state=1).fit(X_train_scaled, y_train)
# Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

#------
# Import an Adaptive Boosting classifier
from sklearn.ensemble import AdaBoostClassifier
# Train the AdaBoostClassifier
clf = AdaBoostClassifier(random_state=1).fit(X_train_scaled, y_train)
# Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

#------- 
# Create the logistic regression classifier model with a random_state of 1
lr_model = LogisticRegression(random_state=1)
# Fit the model to the training data
lr_model.fit(X_train_encoded, y_train)

#------


#------


# ----------------------------------------------------------------
# ----- Metrics
# ----------------------------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score

# ----- Confusion Matrix
# Make predictions on the test data
predictions = classifier.predict(X)
# Create a confusion matrix
print(confusion_matrix(y, predictions, labels = [1,0]))

# -----  Create a classification report
print(classification_report(y, predictions, labels = [1, 0]))

Output:
              precision    recall  f1-score   support

           1       0.74      0.16      0.26      2000
           0       0.90      0.99      0.94     14632

    accuracy                           0.89     16632
   macro avg       0.82      0.58      0.60     16632
weighted avg       0.88      0.89      0.86     16632

# ----- Calculate the balanced accuracy score
# Check the model's balanced accuracy on the test set
y_pred = model.predict(X_test)
print(balanced_accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y, predictions))

# ----- Calculate the balanced accuracy score
# Check the model's balanced accuracy on the test set
y_test_pred = model.predict(X_test_encoded)
print(balanced_accuracy_score(y_test_encoded, y_test_pred))
# Check the model's balanced accuracy on the training set
y_train_pred = model.predict(X_train_encoded)
print(balanced_accuracy_score(y_train_encoded, y_train_pred))


# ----- Predict values with probabilities
# Predict values with probabilities
pred_probas = classifier.predict_proba(X)

# Print the probabilities
pred_probas

# Each prediction includes a prediction for both the 0 class and the 1 class
# We only need the predictions for the 1 class; use a list comprehension to 
# gather the second value from each list
pred_probas_firsts = [prob[1] for prob in pred_probas]

# Print the first 5 probabilities
pred_probas_firsts[0:5]

# ----- roc_auc_score
# Calculate the roc_auc_score
print(roc_auc_score(y, pred_probas_firsts))

# ------- accuracy score
from sklearn.metrics import accuracy_score# Display the accuracy score for the testing dataset
accuracy_score(y_test, testing_predictions)

# ----------------------------------------------------------------
# ----- Overfitting
# ----------------------------------------------------------------
# --------------------------------
# --- example 1
# Create a loop to vary the max_depth parameter
# Make sure to record the train and test scores 
# for each pass.

# Depths should span from 1 up to 40 in steps of 2
depths = range(1, 40, 2)

# The scores dataframe will hold depths and scores
# to make plotting easy
scores = {'train': [], 'test': [], 'depth': []}

# Loop through each depth (this will take time to run)
for depth in depths:
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    scores['depth'].append(depth)
    scores['train'].append(train_score)
    scores['test'].append(test_score)

# Create a dataframe from the scores dictionary and
# set the index to depth
scores_df = pd.DataFrame(scores).set_index('depth')

scores_df.head()

# Plot the scores dataframe with the plot method
scores_df.plot()

# --------------------------------
# --- Example 2
# We overfit! Lets try varying the max depth

models = {'train_score': [], 'test_score': [], 'max_depth': []}

for depth in range(1,10):
    models['max_depth'].append(depth)
    model = RandomForestClassifier(n_estimators=500, max_depth=depth)
    model.fit(X_train_encoded, y_train_encoded)
    y_test_pred = model.predict(X_test_encoded)
    y_train_pred = model.predict(X_train_encoded)

    models['train_score'].append(balanced_accuracy_score(y_train_encoded, y_train_pred))
    models['test_score'].append(balanced_accuracy_score(y_test_encoded, y_test_pred))

models_df = pd.DataFrame(models)

models_df.plot(x='max_depth')

# it looks like the lines start to diverge a lot after 7
# Create and train a RandomForest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth = 7, n_estimators=100)
model.fit(X_train_encoded, y_train_encoded)

y_train_pred = model.predict(X_train_encoded)
print(balanced_accuracy_score(y_train_encoded, y_train_pred))
y_test_pred = model.predict(X_test_encoded)
print(balanced_accuracy_score(y_test_encoded, y_test_pred))


# ----------------------------------------------------------------
# ----- Correlation
# ----------------------------------------------------------------
# Check correlation of columns
df.corr()['Result'].sort_values()
# Plot app_number and Result in a scatter plot
df.plot(kind='scatter', x='app_number', y='Result')

# ----------------------------------------------------------------
# ----- Parameter Tuning
# ----------------------------------------------------------------
# ------------- 14.3.2
# Create three KNN classifiers
from sklearn.neighbors import KNeighborsClassifier
untuned_model = KNeighborsClassifier()
grid_tuned_model = KNeighborsClassifier()
random_tuned_model = KNeighborsClassifier()

## Train a model without tuning
from sklearn.metrics import classification_report
untuned_model.fit(X_train, y_train)
untuned_y_pred = untuned_model.predict(X_test)
print(classification_report(y_test, untuned_y_pred,
                            target_names=target_names))
# Fit the model by using the grid search estimator.
# This will take the KNN model and try each combination of parameters.
grid_clf.fit(X_train, y_train)
# List the best parameters for this dataset
print(grid_clf.best_params_)
# Print the classification report for the best model
grid_y_pred = grid_clf.predict(X_test)
print(classification_report(y_test, grid_y_pred,
                            target_names=target_names))

# -------------14.3.2
# Create the parameter object for the randomized search estimator.
# Try adjusting n_neighbors with values of 1 through 19. 
# Adjust leaf_size by using a range from 1 to 500.
# Include both uniform and distance options for weights.
param_grid = {
    'n_neighbors': np.arange(1,20,2),
    'weights': ['uniform', 'distance'],
    'leaf_size': np.arange(1, 500)
}
param_grid

# Create the randomized search estimator
from sklearn.model_selection import RandomizedSearchCV
random_clf = RandomizedSearchCV(random_tuned_model, param_grid, random_state=0, verbose=3)

# Fit the model by using the randomized search estimator.
random_clf.fit(X_train, y_train)

# List the best parameters for this dataset
print(random_clf.best_params_)

# Make predictions with the hypertuned model
random_tuned_pred = random_clf.predict(X_test)

# Calculate the classification report
print(classification_report(y_test, random_tuned_pred,
                            target_names=target_names))


# ----------------------------------------------------------------
# ----- Imbalanced classification
# ----------------------------------------------------------------
# ------------ RandomUnderSampler    14.3.4
# Import RandomUnderSampler from imblearn
from imblearn.under_sampling import RandomUnderSampler
# Instantiate a RandomUnderSampler instance
rus = RandomUnderSampler(random_state=1)
# Fit the training data to the random undersampler model
X_undersampled, y_undersampled = rus.fit_resample(X_train_scaled, y_train)
# Count distinct values for the resampled target data
y_undersampled.value_counts()

# Instantiate a new RandomForestClassier model
model_undersampled = RandomForestClassifier()
# Fit the undersampled data the new model
model_undersampled.fit(X_undersampled, y_undersampled)
# Predict labels for oversampled testing features
y_pred_undersampled = model_undersampled.predict(X_test_scaled)

# Print classification reports
print(f"Classification Report - Original Data")
print(classification_report(y_test, y_pred))
print("---------")
print(f"Classification Report - Undersampled Data")
print(classification_report(y_test, y_pred_undersampled))

# ------------ RandomOverSampler     14.3.4
# Import RandomOverSampler from imblearn
from imblearn.over_sampling import RandomOverSampler
# Instantiate a RandomOversampler instance
ros = RandomOverSampler(random_state=1)

# Fit the training data to the `RandomOverSampler` model
X_oversampled, y_oversampled = ros.fit_resample(X_train_scaled, y_train)

# Count distinct values
y_oversampled.value_counts()

# Instantiate a new RandomForestClassier model
model_oversampled = RandomForestClassifier()

# Fit the oversampled data the new model
model_oversampled.fit(X_oversampled, y_oversampled)

# Predict labels for oversampled testing features
y_pred_oversampled = model_oversampled.predict(X_test_scaled)

# Print classification reports
print(f"Classification Report - Original Data")
print(classification_report(y_test, y_pred))
print("---------")
print(f"Classification Report - Undersampled Data")
print(classification_report(y_test, y_pred_undersampled))
print("---------")
print(f"Classification Report - Oversampled Data")
print(classification_report(y_test, y_pred_oversampled))


# ------------ ClusterCentroids
# Import ClusterCentroids from imblearn
from imblearn.under_sampling import ClusterCentroids
# Instantiate a ClusterCentroids instance
cc_sampler = ClusterCentroids(random_state=1)


# ------------ SMOTE
# Import SMOTE from imblearn
from imblearn.over_sampling import SMOTE
# Instantiate the SMOTE instance 
# Set the sampling_strategy parameter equal to auto
smote_sampler = SMOTE(random_state=1, sampling_strategy='auto')


# ------------ SMOTEEN
# Import SMOTEEN from imblearn
from imblearn.combine import SMOTEENN
# Instantiate the SMOTEENN instance
smote_enn = SMOTEENN(random_state=1)


# ------------

# ------------

# ------------

# ----------------------------------------------------------------
# ----- 
# ----------------------------------------------------------------
