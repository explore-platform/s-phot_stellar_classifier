import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from astropy.coordinates import SkyCoord
from astropy import units as u


def remove_outliers(data, classes):
    idx = pd.IndexSlice
    clean_data = data.loc[:, idx["Fitted", ["Teff", "Lum"], "Value", :, :, :, :]]
    filter1 = clean_data["Fitted", "Lum"] < 10**6
    filter2 = clean_data["Fitted", "Teff"] < 2.1 * 10**5
    final_filter = [a and b for a, b in zip([i[0] for i in filter2.to_numpy()], [i[0] for i in filter1.to_numpy()])]
    clean_data = data[final_filter]
    clean_classes = classes[final_filter]
    return clean_data, clean_classes


def replace_missing_data(row, method='nan'):
    if method == 'nan':
        return row.replace(['[]', '--', '[', ']'], np.nan)
    

def read_data(dataloc_data, dataloc_labels, num_header_lines = 9):
    delimiter_used = '\t'
    if dataloc_data[-4:] == '.csv':
        delimiter_used = ','
    df = pd.read_csv(dataloc_data, skiprows=num_header_lines, header=None, delimiter=delimiter_used, dtype={0: str})
    classes = pd.read_csv(dataloc_labels, header=None, delimiter=delimiter_used, dtype={0: str})

    classes[0] = classes[0].str.replace(" ", "_")
    # for backwards compatibility (1perc dataset)
    df[0] = df[0].str.rstrip('_')
    # Extract header
    header = []

    with open(dataloc_data, "r") as fp:
        for i, line in enumerate(fp):
            if i < num_header_lines:
                header.append(line.rstrip().split(delimiter_used))
            else:
                break

    return df, classes, header


def custom_train_test_split_2(lab, test_size=0.2, min_class_instances=10):
    np.random.seed(1606421)
    test_count = int(len(lab) * test_size)
    label_indices_dict = {}
    remaining_sample_indices = []
    test_indices = []
    train_indices = []
    possible_classes = np.unique(sum([i.split(", ") for i in lab['class']], []))
    for j in possible_classes:
        label_indices_dict[j] = [i for i, l in enumerate(lab['class']) if j in l.split(", ") and i not in test_indices]
        test_indices.extend(random.sample(label_indices_dict[j], min_class_instances))
    
    remaining_sample_indices.extend([i for i in range(len(lab['class'])) if i not in test_indices])

    test_indices.extend(random.sample(remaining_sample_indices, test_count - len(test_indices)))

    train_indices = []
    train_indices.extend([i for i in remaining_sample_indices if i not in test_indices])

    # shuffle data
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return train_indices, test_indices


def process_data_frame_2(data, classes, header):
    np.random.seed(1606421)
    
    # Replace Missing Data
    data = data.apply(replace_missing_data, axis=1)
    data.iloc[:, 0] = data.iloc[:, 0].astype(str)
    
    # Set Multi-Header in Data
    multi_head = pd.MultiIndex.from_arrays(header, names=["Origin", "Measurement", "Type", "Unit", "Wavelength", "Width", "Alambda"])
    data.columns = multi_head
    data = data.set_index(data.iloc[:, 0].values, drop=True)
    data = data.drop(data.columns[[0]], axis=1)
    # for backwards compatibility (1perc dataset)
    data = data.loc[:, ~data.columns.duplicated()].copy()

    # Set Index For Classes
    classes = classes.set_index(classes.iloc[:, 0].values, drop=True)
    classes = classes.drop(classes.columns[0], axis=1)
    classes.columns = ["class"]

    # Align Indices for Classes and Data
    classes = classes.sort_index(axis=0)
    data = data.sort_index(axis=0)
    assert (data.index.equals(classes.index))

    # Set PySSED inferred fields to NAN, in case of failure
    pyssed_failed = np.array([i[0] for i in data[('Fitted', 'Teff', 'Value')].eq(0).values]
                            and [i[0] for i in data[('Fitted', 'Lum', 'Value')].eq(0).values])
    pyssed_stars_failed = np.where(pyssed_failed)[0]
    pyssed_cols_i = [2, 3]
    pyssed_cols_i.extend(list(range(6, 20)))
    pyssed_cols_i.extend(data.columns.get_loc(i) for i in data.xs(("Model", "Value"), level=("Origin", "Type"), axis=1, drop_level=False).columns)
    data.iloc[pyssed_stars_failed, pyssed_cols_i] = np.nan

    # Add dtype Info to DataFrame
    string_cols = data.xs(("Ancillary", "Source"), level=("Origin", "Type"), axis=1, drop_level=False).columns
    data.loc[:, ~data.columns.isin(string_cols)] = data.loc[:, ~data.columns.isin(string_cols)].astype(float, errors='ignore')

    return data, classes


def classes_to_multilabel(classes):
    mlb = preprocessing.MultiLabelBinarizer()
    encoded_classes = mlb.fit_transform(classes)
    return encoded_classes, mlb


def add_multilabel_as_novel_samples(data, classes):
    # Add Multilabel Stars as New Stars to Data.
    duplicate_stars_X = []
    new_classes = []

    for i, star_name in enumerate(classes.index):
        if type(classes.loc[star_name].values[0]) != str:
            star_class = classes.loc[star_name].values[0][0].split(", ")
        else:
            star_class = classes.loc[star_name].values[0].split(", ")
        classes.iloc[i] = star_class[0]
        if len(star_class) > 1:
            for j, c in enumerate(star_class[1:]):
                duplicate_stars_X.append(data.iloc[i, :].rename(f"{star_name}_{str(j)}"))
                new_classes.append([f"{star_name}_{str(j)}", c])

    # Reindex Newly Added Stars
    new_classes_indexed = pd.DataFrame(new_classes, columns=['star_name', 'class'])
    new_classes_indexed = new_classes_indexed.set_index('star_name', drop=True)
    final_classes = pd.concat([classes, new_classes_indexed], axis=0)
    final_X = pd.concat([data, pd.DataFrame(duplicate_stars_X)], axis=0)

    # Finalize Input Files
    assert(final_X.index.equals(final_classes.index))
    final_classes = final_classes.sort_index(axis=0)
    final_X = final_X.sort_index(axis=0)
    return final_X, final_classes


def convert_to_galactic_coords(data):
    coords = SkyCoord(data[("Adopted", "RA", "Value")].values * u.degree, data[("Adopted", "Dec", "Value")].values * u.degree)
    coord1 = coords.galactic.l / u.degree
    coord2 = coords.galactic.b / u.degree
    data[("Adopted", "RA", "Value")] = coord1
    data[("Adopted", "Dec", "Value")] = coord2
    return data


def transform_Spectra(data, corrected_error=False, corrected_error_2=False):
    idx = pd.IndexSlice
    dereddened = data.loc[:, idx["Dereddened", :, "Value", :, :, :, :]]
    model = data.loc[:, idx["Model", :, "Value", :, :, :, :]]
    phot_error = data.loc[:, idx["Photometry", :, "Error", :, :, :, :]]
    
    if corrected_error:
        raw_values = data.loc[:, idx["Photometry", :, "Value", :, :, :, :]]
        ten_percent = raw_values.multiply(0.1, axis=1)
        tmp_df = pd.concat([ten_percent, phot_error], axis=1)
        sub_error = pd.DataFrame()
        for i, col in enumerate(ten_percent.columns):
            tmp_indexer = list(col)
            tmp_indexer[2] = 'Error'
            sub_error[i] = np.sqrt(tmp_df[[col, tmp_indexer]] ** 2).sum(axis=1).to_numpy()
        # transform_spectra
        dereddened.columns = list(range(len(dereddened.columns)))
        model.columns = list(range(len(model.columns)))
        phot_error = pd.DataFrame(sub_error)
        phot_error.columns = list(range(len(phot_error.columns)))
        phot_error.replace(['0', 0], np.nan, inplace=True)
        transformed_spectra = dereddened.sub(model, fill_value=0, axis=0).div(phot_error.to_numpy(), axis=0)
    elif corrected_error_2:
        raw_values = data.loc[:, idx["Photometry", :, "Value", :, :, :, :]]
        ten_percent = raw_values.multiply(0.1, axis=1)
        tmp_df = pd.concat([ten_percent, phot_error], axis=1)
        sub_error = pd.DataFrame()
        for i, col in enumerate(ten_percent.columns):
            tmp_indexer = list(col)
            tmp_indexer[2] = 'Error'
            sub_error[i] = np.sqrt(tmp_df[[col, tmp_indexer]] ** 2).sum(axis=1).to_numpy()
        # transform_spectra
        dereddened.columns = list(range(len(dereddened.columns)))
        model.columns = list(range(len(model.columns)))
        phot_error = pd.DataFrame(sub_error)
        phot_error.columns = list(range(len(phot_error.columns)))
        phot_error.replace(['0', 0], np.nan, inplace=True)
        #new_error = dereddened.sub(model, fill_value=0, axis=0).div(phot_error.to_numpy(), axis=0)
        transformed_spectra = np.log(dereddened.div(model, fill_value=1, axis=0)).div(np.log(phot_error.to_numpy() + 1), axis=0)
    else:
        # transform_spectra
        dereddened.columns = list(range(len(dereddened.columns)))
        model.columns = list(range(len(model.columns)))
        phot_error.columns = list(range(len(phot_error.columns)))
        transformed_spectra = dereddened.sub(model, fill_value=0, axis=0).div(phot_error.apply(lambda row: row.replace(['0', 0], np.nan), axis=1), fill_value=np.finfo(np.float64).eps, axis=0)
    SpectraPackage = pd.DataFrame(transformed_spectra) #, columns=data.loc[:, idx["Dereddened", :, "Value", :, :, :, :]].columns
    SpectraPackage.columns = data.loc[:, idx["Dereddened", :, "Value", :, :, :, :]].columns
    return SpectraPackage


def merge_duplicate_wavelength_cols(data):
    new_wavelengths = []
    wavelength_signitures = [("Photometry", "Value"), ("Photometry", "Error"), ("Dereddened", "Value"), ("Model", "Value")]
    for wavelength_sig in wavelength_signitures:
        sed_subset = data.xs(wavelength_sig, level=("Origin", "Type"), axis=1, drop_level=False)
        new_sed = []
        for wavelength in sed_subset.columns.get_level_values("Wavelength").unique():
            dupe_data = sed_subset.xs(wavelength, level="Wavelength", axis=1, drop_level=False)
            averaged_dupe = dupe_data.mean(axis=1)
            averaged_dupe = averaged_dupe.rename(dupe_data.columns[0])
            new_sed.append(averaged_dupe)
        new_wavelengths.extend(new_sed)
    for origins in ["Photometry", "Dereddened", "Model"]:
        data = data.drop(origins, axis=1, level="Origin")
    tmp_dat = pd.concat(new_wavelengths, axis=1)
    return pd.concat([data, tmp_dat], axis=1)


def label_encode_data(classes):
    # label_encode categorical output
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(classes['class'].values)
    assert(len(labels) == len(classes))
    return le, labels
