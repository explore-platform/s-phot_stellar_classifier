import optuna
from timeit import default_timer as timer
from optuna import distributions
import explore_commons
import numpy as np
import pandas as pd
from sklearn import metrics, utils, model_selection
import matplotlib.pyplot as plt
from XGBoost_Weighted import XGBClassifier_w


# Set seed for reproducibility
np.random.seed(1606421)

idx = pd.IndexSlice

# Read to dataframe
df, classes, header = explore_commons.read_data('data/ms_augmented/df_plus_ms.dat',
                                                'data/ms_augmented/classes_plus_ms.dat')


# Preprocess df
df, classes = explore_commons.process_data_frame_2(df, classes, header[2:])

# Merge duplicate Wavelengths
df = explore_commons.merge_duplicate_wavelength_cols(df)

# Galactic Coordinate Conversion
df = explore_commons.convert_to_galactic_coords(df)

# Traditionally Most Relevant Variables Only
BasePackage = df.loc[:, idx["Fitted", ["Teff", "Lum"], "Value", :, :, :, :]]

# Physically Irrelevant Variables Only
BiasPackage = df.loc[:, idx["Adopted", ["RA", "Dec", "PMRA", "PMDec", "Distance"], "Value", :, :, :, :]]

# Spectral Measurements Only
SpectraPackage = pd.concat([df.loc[:, idx["Photometry", :, ["Error"], :, :, :, :]],
                            df.loc[:, idx[["Model", "Dereddened"], :, "Value", :, :, :, :]]], axis=1)

# Transformed Spectra
SpectraPackage = explore_commons.transform_Spectra(df, corrected_error_2=True)

# All Physically Relevant Variables
PhysicsPackage = pd.concat([BasePackage, df.loc[:, idx["Adopted", ["E(B-V)", 'logg', '[Fe/H]'], "Value", :, :, :, :]],
                            df.loc[:, idx["Ancillary", "Tspec", "Value", :, :, :, :]], SpectraPackage], axis=1)

# All Relevant Variables
FullPackage = pd.concat([BiasPackage, PhysicsPackage], axis=1)


tune_iter = 2000
avail_threads = -1
all_models = []

# Split Indices
train_indices, test_indices = explore_commons.custom_train_test_split_2(classes, test_size=0.2, min_class_instances=40)

DataPackages = {'Full': FullPackage, 'Physics': PhysicsPackage, 'Spectra': SpectraPackage, 'Bias': BiasPackage, 'Base': BasePackage}

for dat_pack in DataPackages.items():
    start = timer()

    X_train_full = dat_pack[1].iloc[train_indices]
    X_test = dat_pack[1].iloc[test_indices]
    y_test = classes.iloc[test_indices]
    y_train_full = classes.iloc[train_indices]

    y_train_full, mlb = explore_commons.classes_to_multilabel([i for i in y_train_full["class"].str.split(", ")])
    y_test = mlb.transform([i for i in y_test["class"].str.split(", ")])

    X_train_full.columns = range(X_train_full.shape[1])
    X_test.columns = range(X_test.shape[1])

    #X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_full, y_train_full, train_size=0.1)

    def objective(trial, _X, _y):
        n_estimators = trial.suggest_int('n_estimators', high=2000, low=250, step=250)
        #n_estimators = 500
        w0 = trial.suggest_float('w0', high=1.0, low=0.001, step=0.001)
        w1 = trial.suggest_float('w1', high=1.0, low=0.001, step=0.001)
        w2 = trial.suggest_float('w2', high=1.0, low=0.001, step=0.001)
        w3 = trial.suggest_float('w3', high=1.0, low=0.001, step=0.001)
        w4 = trial.suggest_float('w4', high=1.0, low=0.001, step=0.001)
        w5 = trial.suggest_float('w5', high=1.0, low=0.001, step=0.001)
        w6 = trial.suggest_float('w6', high=1.0, low=0.001, step=0.001)
        w7 = trial.suggest_float('w7', high=1.0, low=0.001, step=0.001)
        w8 = trial.suggest_float('w8', high=1.0, low=0.001, step=0.001)
        eta =  trial.suggest_float('eta', high=0.31, low=0.001, step=0.005)
        #eta=0.1
        max_depth = trial.suggest_int('max_depth', high=10, low=3, step=1)
        min_child_weight = trial.suggest_int('min_child_weight', high=20, low=0, step=2)
        subsample = trial.suggest_float('subsample', high=1.0, low=0.2, step=0.1)
        colsample_bytree = trial.suggest_float('colsample_bytree', high=1.0, low=0.3, step=0.1)
        gamma = trial.suggest_float('gamma', high=10.0, low=0.0, step=0.2)

        model = XGBClassifier_w(w0, w1, w2, w3, w4, w5, w6, w7, w8, n_estimators=n_estimators, eta=eta, max_depth=max_depth,
                                min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
                                gamma=gamma, tree_method='hist', random_state=1606421, verbosity=0)

        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=1606421)
        f1_mac_scorer = metrics.make_scorer(metrics.f1_score, average='macro')

        cv_results = model_selection.cross_val_score(model, _X, _y, scoring=f1_mac_scorer, cv=kf)
        #model.fit(X_train,y_train)
        #y_pred = model.predict(X_val)
        #f1 = metrics.f1_score(y_pred, y_val, average='macro')
        #return f1

        # https://medium.com/@walter_sperat/using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451
        return np.min([np.mean(cv_results), np.median(cv_results)])

    # optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full), n_trials=tune_iter, n_jobs=avail_threads)
    trial = study.best_trial

    print(trial.params)
    print(trial.values)
    xgb_best = XGBClassifier_w(**trial.params, verbosity=0)
    xgb_best.fit(X_train_full, y_train_full)
    y_pred = xgb_best.predict(X_test)
    end=timer()
    with open(f'output_{dat_pack[0]}.txt', 'w') as f:
        f.write(f'''
{trial.params}
{trial.values}
{str(metrics.accuracy_score(y_pred, y_test))}
{str(metrics.f1_score(y_pred, y_test, average='macro'))}
Time taken: {str(end-start)}''')