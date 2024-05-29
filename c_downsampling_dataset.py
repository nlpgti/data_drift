import os
import warnings

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def filter_by_rating_dataset(dataset, mode):
    if mode == 4:
        return dataset.loc[dataset['Rating'] < 5]
    elif mode == 5:
        return dataset.loc[dataset['Rating'] == 5]


def balance_dataset(dataset):
    under_sampler = RandomUnderSampler(random_state=1)
    dataset, _ = under_sampler.fit_resample(dataset, dataset["Label"])
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.sort_values(by=['Timestamp'], ascending=True)
    dataset = dataset.reset_index(drop=True)
    return dataset


def balance_by_rating(path, mode):
    dataset = pd.read_csv(path, sep=",", engine="pyarrow")
    dataset_rating = filter_by_rating_dataset(dataset, mode)
    dataset_rating = balance_dataset(dataset_rating)
    dataset_rating.to_csv("datasets/balanced_" + str(mode) + ".csv", index=False,
                          header=True)


def balance_and_merge_datasets():
    list_datasets = []
    # The 5-star rating is the majority category, we balance the data set in two steps
    balance_by_rating("datasets/text_preprocess_dataset_incremental.csv", 4)
    balance_by_rating("datasets/text_preprocess_dataset_incremental.csv", 5)
    dataset1 = pd.read_csv("datasets/balanced_4.csv", sep=",", engine="pyarrow")
    dataset2 = pd.read_csv("datasets/balanced_5.csv", sep=",", engine="pyarrow")
    list_datasets.append(dataset1)
    list_datasets.append(dataset2)
    result = pd.concat(list_datasets, ignore_index=True)
    result = result.sort_values(by=['Timestamp'], ascending=True)
    result.reset_index(drop=True, inplace=True)
    result.to_csv("datasets/dataset_balanced.csv", index=False, header=True)
    print(result["Label"].value_counts())

    # Prepare data set for parallel execution
    dataset_splited = np.array_split(result, 20)
    for index, df in enumerate(dataset_splited):
        df.to_csv("datasets/dataset_balanced" + "_" + str(index) + ".csv", index=False, header=True)


if __name__ == '__main__':
    balance_and_merge_datasets()
