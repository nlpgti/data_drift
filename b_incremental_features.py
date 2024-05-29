import os
import time
import warnings
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def avg_maximum_cumsum(dataset_per_user, feature_name, calc_name):
    dataset_per_user[feature_name + '_avg' + "_" + calc_name] = dataset_per_user[feature_name].cumsum()
    dataset_per_user[feature_name + '_avg' + "_" + calc_name] = dataset_per_user[
                                                                    feature_name + '_avg' + "_" + calc_name] / \
                                                                dataset_per_user['tweet_num' + "_" + calc_name]
    dataset_per_user[feature_name + '_higher' + "_" + calc_name] = np.where(
        dataset_per_user[feature_name + '_avg' + "_" + calc_name] > dataset_per_user[feature_name], False, True)
    return dataset_per_user


def lambda_funtions(x, calc_name):
    if x['was_spam' + "_" + calc_name] > 1:
        x['was_spam' + "_" + calc_name] = 1
    else:
        x['was_spam' + "_" + calc_name] = 0
    x['year_week' + "_" + calc_name] = str(date.fromtimestamp(x["Timestamp"]).year) + str(
        date.fromtimestamp(x["Timestamp"]).isocalendar()[1])
    x["polarity_deviation" + "_" + calc_name] = abs(x["polarity"] - (x["Rating"] + 1) / 2 * 5)
    return x


def target_category_to_num(x):
    if x == "fake":
        return 1
    elif x == "nofake":
        return 0


def incremental_elements_by_param(data_to_calc):
    dataset = data_to_calc["dataset"]
    element_under_analysis = data_to_calc["calc_element"]
    param = data_to_calc["param"]
    calc_name = data_to_calc["calc_name"]
    more_than_one_element = data_to_calc["more_than_one_element"]
    if more_than_one_element:
        dataset_per_element = dataset[(dataset[param["element1"]] == element_under_analysis["element1"]) & (
                dataset[param["element2"]] == element_under_analysis["element2"])]
        dataset_per_element = dataset_per_element.sort_values(by=[param["element1"], param["element2"], 'Timestamp'])
    else:
        dataset_per_element = dataset[dataset[param] == element_under_analysis]
        dataset_per_element = dataset_per_element.sort_values(by=[param, 'Timestamp'])

    dataset_per_element.reset_index(drop=True, inplace=True)

    if len(dataset_per_element) > 0:

        #Feature 82 in Table 4
        dataset_per_element['tweet_num' + "_" + calc_name] = range(1, 1 + len(dataset_per_element))
        dataset_per_element['Label_num' + "_" + calc_name] = dataset_per_element["Label"].apply(
            lambda x: target_category_to_num(x))
        # Feature 83 in Table 4
        dataset_per_element['was_spam' + "_" + calc_name] = dataset_per_element["Label_num" + "_" + calc_name].cumsum()
        dataset_per_element = dataset_per_element.apply(lambda x: lambda_funtions(x, calc_name), axis=1)

        list_features_to_avg_maximum = ['Rating', 'Word_count', 'pos_prop_ADJ', 'pos_prop_ADV',
                                'char_counts', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise',
                                'flesch_reading_ease', 'pos_prop_INTJ', 'mcalpine_eflaw', 'polarity',
                                'pos_prop_PUNCT', 'reading_time', 'pos_prop_VERB', 'urls',
                                'pos_prop_NOUN', 'pos_prop_PRON', 'difficult_words_count',
                                "polarity_deviation" + "_" + calc_name]

        for z in list_features_to_avg_maximum:
            dataset_per_element = avg_maximum_cumsum(dataset_per_element, z, calc_name)

        index_duplicated_yearweek = pd.Index(dataset_per_element["year_week" + "_" + calc_name]).duplicated()
        dataset_per_element['week_num' + "_" + calc_name] = (~index_duplicated_yearweek).astype(int)
        # Feature 84 in Table 4
        dataset_per_element['week_num' + "_" + calc_name] = dataset_per_element["week_num" + "_" + calc_name].cumsum()
        # Feature 85 in Table 4
        dataset_per_element['tweet_freq_week' + "_" + calc_name] = dataset_per_element['tweet_num' + "_" + calc_name] / \
                                                                   dataset_per_element['week_num' + "_" + calc_name]

        return dataset_per_element
    else:
        return None


def incremental_features():
    dataset = pd.read_csv("datasets/text_preprocess_dataset.csv", engine="pyarrow")

    init_time = time.time()
    list_elements = []
    for z in dataset['User_id'].unique():
        list_elements.append({"dataset": dataset, "calc_element": z, "calc_name": "user", "param": "User_id",
                              "more_than_one_element": False})
    print("Pool started")
    p = Pool()
    list_data = p.map(incremental_elements_by_param, list_elements)
    p.close()
    p.join()
    print("Pool finished")
    print(time.time() - init_time)
    dataset = pd.concat(list_data)
    del dataset["Label_num_" + "user"]
    del dataset["year_week_" + "user"]

    list_elements = []
    for z in dataset['Product_id'].unique():
        list_elements.append({"dataset": dataset, "calc_element": z, "calc_name": "product", "param": "Product_id",
                              "more_than_one_element": False})
    print("Pool started")
    #Features 83-139 Table 5
    p = Pool()
    list_data = p.map(incremental_elements_by_param, list_elements)
    p.close()
    p.join()
    print("Pool finished")
    print(time.time() - init_time)
    dataset = pd.concat(list_data)
    del dataset["Label_num_" + "product"]
    del dataset["year_week_" + "product"]

    list_elements = []
    for product in dataset['Product_id'].unique():
        for z in range(1, 6):
            list_elements.append(
                {"dataset": dataset, "calc_element": {"element1": product, "element2": z}, "calc_name": "rating",
                 "param": {"element1": "Product_id", "element2": "Rating"},
                 "more_than_one_element": True})
    print("Pool started")
    # Features 140-177 Table 5
    p = Pool()
    list_data = p.map(incremental_elements_by_param, list_elements)
    p.close()
    p.join()
    list_data = [x for x in list_data if x is not None]
    print("Pool finished")
    print(time.time() - init_time)
    dataset = pd.concat(list_data)
    del dataset["Label_num_" + "rating"]
    del dataset["year_week_" + "rating"]

    columns_saved = ['User_id', 'Product_id', 'Rating', 'Date', 'Review',
                     'Timestamp', 'Year', 'Word_count', 'pos_prop_ADJ', 'pos_prop_ADV',
                     'char_counts', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise',
                     'flesch_reading_ease', 'pos_prop_INTJ', 'mcalpine_eflaw', 'polarity',
                     'pos_prop_PUNCT', 'reading_time', 'pos_prop_VERB', 'urls',
                     'pos_prop_NOUN', 'pos_prop_PRON', 'difficult_words_count',
                     'text_preprocessed', 'Label']

    for type_calc in ["user", "product", "rating"]:
        columns_saved = columns_saved + list(dataset.columns[dataset.columns.str.contains(type_calc)])

    columns_saved.remove("was_spam_product")
    columns_saved.remove("week_num_product")
    columns_saved.remove("tweet_freq_week_product")
    columns_saved.remove("tweet_num_product")

    columns_saved.remove("was_spam_rating")
    columns_saved.remove("week_num_rating")
    columns_saved.remove("tweet_freq_week_rating")
    columns_saved.remove("tweet_num_rating")

    dataset_result = dataset[columns_saved]
    dataset_result.to_csv("datasets/text_preprocess_dataset_incremental.csv", index=False, header=True)


if __name__ == '__main__':
    incremental_features()
