# Load and pre-process the AllState training data.
# Each row consists of one index, 116 categorical predictors, 14 continuous
# predictors, and one continuous response variable called loss.
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Train and test size settings
N_load     = 30000
f_test     = 0.50
N_train    = 8000

# RandomForestRegressor settings:
nTrees     = 250
mxDepth    = 20
mnSleaf    = 5
pp         = 2
runOptions = '_{0:1d}_mxft1o3_{1:02d}K'.format(pp, N_train/1000)

fName      = '{0}_{1}_{2}_{3}'.format(runOptions, mxDepth, mnSleaf, nTrees)

# Preprocessing routine
def preProcess1(df):
    """
    Takes data frame df as input, one-hot encodes the categorical variables,
    eliminates binary variables with variance below 0.95*(1-0.95)=0.0475,
    and eliminates one binary variable for each categorical variable that
    is not affected by the variance filter (to avoid the dummy variable trap).
    """
    # Make a dictionary of the number of levels for each categorical feature.
    catdict  = {key: 0 for key in df.columns.values if key[:3]=="cat"}
    for var in catdict.keys():
        catdict[var] = len(df[var].unique())
    print("Total number of categorical feature levels: {0}".format(sum(catdict.values())))

    # Convert categorical variables into subsets of binary ones.
    df = pd.get_dummies( df, drop_first=False)
    print("Shape of data frame after categorical feature conversion: {0}".format(df.shape))

    # Eliminate binary features with low variance.
    cats     = [feature for feature in df.columns.values if feature[:3]=="cat"]
    conts    = [feature for feature in df.columns.values if feature[:4]=="cont"]
    prob     = 0.95
    binvar   = prob * (1.0-prob)
    sel      = VarianceThreshold(threshold=binvar)
    sel.fit(df[cats])
    retain   = sel.get_support(indices=True)
    features = [cats[ind] for ind in retain] + conts + ["loss"]
    df       = df[features]
    print("Shape of data frame after variance filter: {0}".format(df.shape))

    # Eliminate one dummy binary per category not affected by the low-variance filter.
    remove = []
    for key,nlevels in catdict.items():
        binlist = [feature for feature in features if key+"_" in feature]
        if len(binlist) == nlevels:
            remove.append(binlist[0])
    keep   = [feature for feature in features if feature not in remove]
    df     = df[keep]
    print("Shape of data frame after dummy elimination: {0}".format(df.shape))

    # All done!
    return df

# Load the training data set (set nrows=None to load the entire set).
df0 = pd.read_csv("data/train.csv", delimiter=",", header=0, index_col=0, nrows=N_load)
print("Shape of training data frame: %s" %(df0.shape,))

if pp==1:
    df2 = preProcess1(df0)
elif pp==2:
    df2 = preProcess2(df0)

# Convert pandas dataframe to its numpy representation, separating out the predictors.
predictors = [item for item in df2.columns.values if item!="loss"]
X_all      = df2.as_matrix(columns=predictors)
y_all      = df2["loss"]
print('Shape of X_all:         {0}, y_all:         {1}'.format(X_all.shape, y_all.shape))

# Split data set into training and testing sets.
X_train_total, X_test, y_train_total, y_test = train_test_split(X_all, y_all, test_size=f_test, random_state=0)
print('Shape of X_train_total: {0}, y_train_total: {1}'.format(X_train_total.shape, y_train_total.shape))

# Only use a subset of the training set
X_train = X_train_total[:N_train,:]
y_train = y_train_total[:N_train]
print('Shape of X_train:       {0}, y_train:       {1}'.format(X_train.shape, y_train.shape))
print('Shape of X_test:        {0}, y_test:        {1}'.format(X_test.shape, y_test.shape))

# Instantiate a random forest regression model, using Mean Absolute Error as criterion.
rfr = RandomForestRegressor(n_estimators=nTrees, max_depth=mxDepth, min_samples_leaf=mnSleaf, criterion="mae", min_samples_split=2, max_features=0.3333, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=0, verbose=0)

# Fit to training data.
rfr.fit(X_train, y_train)

# Save fitted random forest model to file.
path_to_file = 'fitted_models/rfr'+fName+'.pkl'
try:
    joblib.dump(rfr, path_to_file)
    print('\nRandom forest model saved to {0}'.format(path_to_file))
except:
    print('\nError saving file {0}'.format(path_to_file))

# Make predictions on train and test set and compute evaluation metric (MAE).
y_test_pred  = rfr.predict(X_test)
y_train_pred = rfr.predict(X_train)
mae_train    = mean_absolute_error(y_train, y_train_pred)
mae_test     = mean_absolute_error(y_test, y_test_pred)
dmae         = mae_test - mae_train
r2_train     = r2_score(y_train, y_train_pred)
r2_test      = r2_score(y_test, y_test_pred)
print('\nResults:')
print('                               Train      Test')
print('Number of instances =         {0:6d},   {1:6d}'.format(len(y_train), len(y_test)))
print('Mean y_true =               {0:8.2f}, {1:8.2f}'.format(y_train.mean(), y_test.mean()))
print('Mean y_pred =               {0:8.2f}, {1:8.2f}'.format(y_train_pred.mean(), y_test_pred.mean()))
print('Minimum y_pred =            {0:8.2f}, {1:8.2f}'.format(y_train_pred.min(), y_test_pred.min()))
print('Maximum y_pred =            {0:8.2f}, {1:8.2f}'.format(y_train_pred.max(), y_test_pred.max()))
print('Standard deviation y_true = {0:8.2f}, {1:8.2f}'.format(y_train.std(), y_test.std()))
print('Standard deviation y_pred = {0:8.2f}, {1:8.2f}'.format(y_train_pred.std(), y_test_pred.std()))
print('R^2 =                       {0:8.2f}, {1:8.2f}'.format(r2_train, r2_test))
print('MAE =                       {0:8.2f}, {1:8.2f}, diff: {2:8.2f}'.format(mae_train, mae_test, dmae))

# Plot feature importances
importances = rfr.feature_importances_
indices     = np.argsort(importances)[::-1]
n_features  = min(40,indices.size)
bins        = np.arange(n_features)
x_labels    = np.array(predictors)[indices][:n_features]
maxHeight   = 1.10*max(importances)
fig, axes   = plt.subplots(nrows=1, ncols=1, figsize=(20, 7))
axes.bar(bins, importances[indices][:n_features], align="center", color="lightblue", alpha=0.5)
axes.set_xticks(bins)
axes.set_xticklabels(x_labels, ha="right", rotation=45., fontsize=14)
axes.set_xlim([-0.5,bins.size-0.5])
axes.set_xlabel("Features", fontsize=20)
axes.set_ylim([0.0, maxHeight])
axes.set_ylabel("Importances", fontsize=20)
axes.text(0.79, 0.80, 'Number of trees: {2}\nMax. depth: {0}\nMin. samples/leaf: {1}'.format(mxDepth, mnSleaf, nTrees), fontsize=20, transform=axes.transAxes)
fig.savefig('FeatureImportances'+fName+'.png', dpi=200, bbox_inches='tight')
