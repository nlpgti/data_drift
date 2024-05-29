import os
import time
import warnings

import pandas as pd
from river import feature_selection
from river import stream

from utils.machine_learning_utils import get_selected_params, n_grams, get_new_model, drift_detection
from utils.utils import print_experiment_metrics, run_in_parallel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def experiment_run(config_exp_params):
    n_data_set = None

    cold_start = 500
    init_static = 0
    end_static = cold_start
    len_dynamic = cold_start

    init_value = time.time()
    verbose = config_exp_params["verbose"]
    model_hyper_params = config_exp_params["model_hyper_params"]
    model_name = config_exp_params["model_type"]
    list_params_eval = config_exp_params["list_params_eval"]
    data_drift_analysis = config_exp_params["data_drift_analysis"]
    is_hyperparam_calculation = config_exp_params["is_hyperparam_calculation"]
    data_hyper_params_range = config_exp_params["data_hyper_params_range"]

    y = config_exp_params["y"][:n_data_set]
    x_csv = config_exp_params["X_csv"][:n_data_set]

    x = get_selected_params(x_csv, list_params_eval)

    x_n_grams = pd.DataFrame()
    if not is_hyperparam_calculation:
        x_n_grams, list_n_grams = n_grams(config_exp_params["X"][:n_data_set], max_df_in=config_exp_params["max_df_in"],
                                          min_df_in=config_exp_params["min_df_in"],
                                          ngram_range_in=config_exp_params["ngram_range_in"], max_features_in=None,
                                          cold_start=cold_start)
        x_n_grams = pd.DataFrame(x_n_grams.toarray(), columns=list_n_grams)

    selector = feature_selection.VarianceThreshold()

    model_to_analyse = get_new_model(model_name, model_hyper_params)
    classifier_model = {"model": model_to_analyse, "elements": 0}

    list_y_pred = []
    list_y = []

    index = 0
    x_train = x
    y_train = y
    list_x_river = []
    count_dd = 0
    list_evaluation = []

    for x_river, y_river in stream.iter_pandas(x_train, y_train):
        if selector is not None:
            x_river = selector.learn_one(x_river).transform_one(x_river)

        y_pred = classifier_model["model"].predict_one(x_river)

        if classifier_model["elements"] > 0:
            if y_pred is not None:
                list_y_pred.append(y_pred)
                list_y.append(y_river)
                list_x_river.append(x_river)
            else:
                list_y_pred.append("nofake")
                list_y.append(y_river)
                list_x_river.append(x_river)
        else:
            list_y_pred.append("nofake")
            list_y.append(y_river)
            list_x_river.append(x_river)

        try:
            was_retrained = False
            model_aux = None
            if data_drift_analysis:
                model_aux, init_static, end_static, len_dynamic, was_retrained, values_graph = drift_detection(
                    cold_start, init_static, end_static, len_dynamic, index, x_n_grams, list_x_river,
                    list_y, list_y_pred, model_name, model_hyper_params, data_hyper_params_range)
                if values_graph is not None:
                    list_evaluation.append(values_graph)
            if was_retrained:
                classifier_model["model"] = model_aux
                count_dd = count_dd + 1
            else:
                classifier_model["model"] = classifier_model["model"].learn_one(x_river, y_river)
            classifier_model["elements"] = classifier_model["elements"] + 1
        except Exception as e:
            print(e)

        index = index + 1

    config_exp_params["X_csv"] = config_exp_params["y"] = config_exp_params["X"] = ""
    metrics_values = print_experiment_metrics(model_name, config_exp_params, list_y, list_y_pred,
                                              time.time() - init_value)

    dataset_to_graph = pd.DataFrame(list_evaluation)
    dataset_to_graph.to_csv("datasets/dataset_to_graph.csv", index=False, header=True)
    if is_hyperparam_calculation:
        return metrics_values
    else:
        if verbose:
            print(metrics_values["latex"])
        return list_y, list_y_pred, count_dd


def prepare_dataset(num_samples, path):
    x_csv = pd.read_csv(path, sep=",", engine="pyarrow")
    x_csv = x_csv[:num_samples]
    x_csv = x_csv.sort_values(by=["Timestamp"], ascending=True)
    x_csv = x_csv.reset_index(drop=True)
    x_csv["text_preprocessed"].fillna("", inplace=True)
    x_csv.fillna(0, inplace=True)
    y = x_csv["Label"]
    x = x_csv["text_preprocessed"]

    return x, y, x_csv


def config_analysis(model_name, num_samples, path, data_drift_analysis, verbose=False):
    x, y, x_csv = prepare_dataset(num_samples=num_samples, path=path)

    list_params_eval = ['pos_prop_ADJ', 'pos_prop_ADV', 'char_counts',
                        'difficult_words_count', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise',
                        'flesch_reading_ease', 'pos_prop_INTJ', 'mcalpine_eflaw', 'pos_prop_NOUN',
                        'polarity', 'pos_prop_PRON', 'pos_prop_PUNCT', 'Rating', 'reading_time',
                        'urls', 'pos_prop_VERB','Word_count' ]

    for type_calc in ["user", "product", "rating"]:
        list_params_eval = list_params_eval + list(x_csv.columns[x_csv.columns.str.contains(type_calc)])

    model_hyper_params = []
    hyper_params_range = {}

    if model_name == "arfc":
        model_hyper_params = {'n_models': 1, 'max_features': 50, 'lambda_value': 50}
        hyper_params_range["n_models"] = [1, 50]
        hyper_params_range["max_features"] = [10, 100]
        hyper_params_range["lambda_value"] = [6, 100]

    elif model_name == "hatc":
        model_hyper_params = {'max_depth': 1, 'tie_threshold': 0.0005, 'max_size': 1, 'remove_poor_attrs': True}
        hyper_params_range["max_depth"] = [1, 50, 100]
        hyper_params_range["tie_threshold"] = [0.0005, 0.5, 10, 100]
        hyper_params_range["max_size"] = [1, 50, 100]
        hyper_params_range["remove_poor_attrs"] = [True, False]

    elif model_name == "htc":
        model_hyper_params = {'max_depth': 1, 'tie_threshold': 0.0005, 'max_size': 1, 'remove_poor_attrs': True}
        hyper_params_range["max_depth"] = [1, 50, 100]
        hyper_params_range["tie_threshold"] = [0.0005, 0.5, 10, 100]
        hyper_params_range["max_size"] = [1, 50, 100]
        hyper_params_range["remove_poor_attrs"] = [True, False]

    one_element = {'X': x, 'y': y, "X_csv": x_csv, 'max_df_in': 0.7, 'min_df_in': 0.1, 'ngram_range_in': (1, 2),
                   "model_hyper_params": model_hyper_params, "model_type": model_name,
                   'list_params_eval': list_params_eval,
                   "threshold": 0.0,
                   "data_drift_analysis": data_drift_analysis,
                   "is_hyperparam_calculation": False,
                   "data_hyper_params_range": hyper_params_range,
                   "verbose": verbose
                   }
    return one_element


def run_parallel_experiments(model, num_samples, data_drift_analysis, verbose):
    list_pool = []
    for z in range(20):
        list_pool.append(config_analysis(model_name=model, num_samples=num_samples,
                                         path="datasets/dataset_balanced_" + str(z) + ".csv",
                                         data_drift_analysis=data_drift_analysis, verbose=verbose))
    init_value = time.time()
    list_data = run_in_parallel(experiment_run, list_pool)
    list_y = []
    list_y_pred = []
    list_drift = []
    for z in list_data:
        list_y.extend(z[0])
        list_y_pred.extend(z[1])
        list_drift.append(z[2])
    result = print_experiment_metrics(model, "", list_y, list_y_pred, time.time() - init_value)
    print(result["latex"])


def run_one_experiment(model, num_samples, data_drift_analysis, verbose):
    config_exp_params = config_analysis(model_name=model, num_samples=num_samples,
                                        path="datasets/dataset_balanced.csv",
                                        data_drift_analysis=data_drift_analysis, verbose=verbose)
    experiment_run(config_exp_params)


if __name__ == '__main__':
    # Scenario 1
    run_one_experiment("htc", None, data_drift_analysis=False, verbose=True)
    # Scenario 2
    # run_parallel_experiments("htc", None, data_drift_analysis=False, verbose=False)
    # Scenario 3
    # run_parallel_experiments("htc", None, data_drift_analysis=True, verbose=False)
    # Scenario 4
    # run_one_experiment("arfc", None, data_drift_analysis=True, verbose=True)
