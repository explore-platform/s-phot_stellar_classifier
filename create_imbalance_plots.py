import explore_commons
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
import xgboost as xgb

# Set seed for reproducibility
np.random.seed(1606421)

idx = pd.IndexSlice

# Read to dataframe
df, classes, header = explore_commons.read_data('data/random_10perc/10percent_full.dat',
                                                'data/random_10perc/10percent_labels.txt')

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

all_models = []

# Split Indices
train_indices, test_indices = explore_commons.custom_train_test_split_2(classes, test_size=0.2, min_class_instances=40)

X_train = FullPackage.iloc[train_indices]
X_test = FullPackage.iloc[test_indices]
y_test = classes.iloc[test_indices]
y_train = classes.iloc[train_indices]

X_train.columns = range(X_train.shape[1])
X_test.columns = range(X_test.shape[1])

X_train, y_train = explore_commons.add_multilabel_as_novel_samples(X_train, y_train)
le, labs_train = explore_commons.label_encode_data(y_train)
X_test, y_test = explore_commons.add_multilabel_as_novel_samples(X_test, y_test)
y_labs = le.transform(y_test)

print('''
Output:
Missingness Level, Star Type, F1 for Star, Macro without Star
''')
for star_class in range(0, 9):
    print(le.inverse_transform([star_class]))
    for missingness in np.arange(0, 1.05, 0.05):
        samples_of_interest = np.where(labs_train == star_class)[0]
        samples_to_drop = np.random.choice(samples_of_interest, size=int(missingness*len(samples_of_interest)), replace=False)
        X_class_miss = X_train.drop(X_train.iloc[samples_to_drop].index, axis=0, inplace=False)
        y_class_miss = np.delete(labs_train, samples_to_drop)
        clf = xgb.XGBClassifier(n_estimators=500, eta=0.1, random_state=1606421, tree_method='hist',
                            num_class=9)

        # The class is now 100 percent missing: encode again since only 8 classes exist
        if missingness > 0.95:
            double_encoder = preprocessing.LabelEncoder()
            double_encoded = double_encoder.fit_transform(y_class_miss)
            clf.fit(X_class_miss, double_encoded, verbose=False)
            y_pred = clf.predict(X_test)
            y_pred = double_encoder.inverse_transform(y_pred)
        else:
            clf.fit(X_class_miss, y_class_miss, verbose=False)
            y_pred = clf.predict(X_test)

        f1 = []
        class_report = metrics.classification_report(y_labs, y_pred)
        for i in range(len(le.classes_)):
            metric_results = class_report.splitlines()[2+i].split()
            f1.append(eval(metric_results[3]))

        # plot 1 info
        result_1 = f1[star_class]
        # plot 2 info
        result_2 = f1[:star_class] + f1[star_class+1:]
        print(f"{missingness}, {le.inverse_transform([star_class])[0]}, {str(result_1)}, {str(np.mean(result_2))}")
