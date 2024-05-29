import os
import warnings
import pandas as pd
from river import ensemble
from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from utils.hyper_params_utils import hyperparameter_hatc, \
    hyperparameters_arfc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def get_selected_params(x_train, list_params_eval):
    x_train_counts_aux = pd.DataFrame()
    for param in list_params_eval:
        x_train[param] = x_train[param] * 1
        x_train_counts_aux[param] = x_train[param]

    return x_train_counts_aux


def n_grams(x, max_df_in, min_df_in, ngram_range_in, max_features_in, cold_start):
    count_vect_word = CountVectorizer(
        analyzer='word',
        lowercase=True,
        min_df=min_df_in,
        max_df=max_df_in,
        ngram_range=ngram_range_in,
        max_features=max_features_in,
    )
    count_vect_word.fit_transform(x[0:cold_start])
    x_train_counts = FeatureUnion([("CountVectorizerWords", count_vect_word)
                                   ]).transform(x)

    list_n_grams = count_vect_word.get_feature_names_out()
    return x_train_counts, list_n_grams


def get_new_model(model_name, model_hyper_params):
    model_to_analyse = None
    if model_name == "arfc":
        model_to_analyse = ensemble.AdaptiveRandomForestClassifier(n_models=model_hyper_params["n_models"],
                                                                   max_features=model_hyper_params["max_features"],
                                                                   lambda_value=model_hyper_params["lambda_value"],
                                                                   seed=1)
    elif model_name == "hatc":
        model_to_analyse = HoeffdingAdaptiveTreeClassifier(max_depth=model_hyper_params["max_depth"],
                                                           tie_threshold=model_hyper_params["tie_threshold"],
                                                           max_size=model_hyper_params["max_size"],
                                                           remove_poor_attrs=model_hyper_params["remove_poor_attrs"],
                                                           seed=1)
    elif model_name == "htc":
        model_to_analyse = HoeffdingTreeClassifier(max_depth=model_hyper_params["max_depth"],
                                                   tie_threshold=model_hyper_params["tie_threshold"],
                                                   max_size=model_hyper_params["max_size"],
                                                   remove_poor_attrs=model_hyper_params["remove_poor_attrs"])

    return model_to_analyse


def get_new_hyper_params(data):
    model_name = data["model_name"]
    list_iterations = data["list_iterations"]
    z = data["z"]
    index = data["index"]
    len_dynamic = data["len_dynamic"]
    list_x_river = data["list_x_river"]
    y_original = data["y_original"]

    model_hyper_params = {}
    if model_name == "arfc":
        model_hyper_params["n_models"] = list_iterations[z][0]
        model_hyper_params["max_features"] = list_iterations[z][1]
        model_hyper_params["lambda_value"] = list_iterations[z][2]
    elif model_name == "hatc" or model_name == "htc":
        model_hyper_params["max_depth"] = list_iterations[z][0]
        model_hyper_params["tie_threshold"] = list_iterations[z][1]
        model_hyper_params["max_size"] = list_iterations[z][2]
        model_hyper_params["remove_poor_attrs"] = list_iterations[z][3]

    model_aux = get_new_model(model_name, model_hyper_params)
    list_y_pred_aux = []
    for index_value in range(index - len_dynamic, index):
        try:
            y_pred_aux = model_aux.predict_one(list_x_river[index_value])
            if y_pred_aux is None:
                list_y_pred_aux.append("nofake")
            else:
                list_y_pred_aux.append(y_pred_aux)

            model_aux = model_aux.learn_one(list_x_river[index_value], y_original[index_value])
        except Exception as e:
            print("Error evaluation model: " + str(e))
    acc_aux = accuracy_score(y_original[index - len_dynamic:index], list_y_pred_aux)

    return {"acc_aux": acc_aux, "model_hyper_params": model_hyper_params}


def get_p_value(features, vect_init, vect_second):
    obs = pd.DataFrame([vect_init, vect_second], columns=features)
    obs = obs.drop(columns=obs.columns[(obs < 6).any() > 0])
    stat, p, dof, expected = chi2_contingency(obs)
    return p


def drift_detection(cold_start, init_static, end_static, len_dynamic, index, x, list_x_river, y_original, y_pred,
                    model_name, last_hyper_params, data_hyper_params_range):
    index = index + 1
    was_retrained = False
    model_return = None
    values_graph = None

    if index > cold_start:
        static_window = x[init_static:end_static]

        dynamic_window = x[index - len_dynamic:index]

        acc_static = accuracy_score(y_original[init_static:end_static], y_pred[init_static:end_static])
        y_original_dynamic = y_original[index - len_dynamic:index]
        y_pred_dynamic = y_pred[index - len_dynamic:index]
        acc_dynamic = accuracy_score(y_original_dynamic,
                                     y_pred_dynamic)

        v_static = static_window.sum(axis=0)
        v_dynamic = dynamic_window.sum(axis=0)
        p_value = get_p_value(x.columns, v_static, v_dynamic)

        if p_value >= 0.5:
            if len_dynamic <= 2000:
                len_dynamic = len_dynamic + 1
        elif p_value <= 0.1:
            if len_dynamic > cold_start:
                len_dynamic = len_dynamic - 1

        if abs(acc_static - acc_dynamic) >= 0.05 and p_value <= 0.05 or index == cold_start + 1:
            was_retrained = True
            init_static = index - len_dynamic
            end_static = index

            list_iterations = []
            if model_name == "arfc":
                list_iterations = hyperparameters_arfc(data_hyper_params_range)
            elif model_name == "hatc" or model_name == "htc":
                list_iterations = hyperparameter_hatc(data_hyper_params_range)

            list_data = []
            list_to_evaluation = []

            for z in range(len(list_iterations)):
                data = {"model_name": model_name,
                        "list_iterations": list_iterations,
                        "z": z,
                        "index": index,
                        "len_dynamic": len_dynamic,
                        "list_x_river": list_x_river,
                        "y_original": y_original
                        }
                list_to_evaluation.append(data)

            for configuration in list_to_evaluation:
                list_data.append(get_new_hyper_params(configuration))

            newlist = sorted(list_data, key=lambda d: d['acc_aux'], reverse=True)
            hyper_selected = newlist[0]["model_hyper_params"]

            if last_hyper_params != hyper_selected:
                model_aux = get_new_model(model_name, hyper_selected)
                for index_value in range(index - len_dynamic, index):
                    try:
                        if model_name == "alma":
                            if y_original[index_value] == "fake":
                                y_aux_river = 1
                            else:
                                y_aux_river = 0
                            model_aux = model_aux.learn_one(list_x_river[index_value], y_aux_river)
                        else:
                            model_aux = model_aux.learn_one(list_x_river[index_value], y_original[index_value])
                    except Exception as e:
                        print("Error training new model: " + str(e))

                model_return = model_aux
            else:
                was_retrained = False

        values_graph = {
            "index": index,
            "acc_static": acc_static,
            "acc_dinamic": acc_dynamic,
            "ada": abs(acc_static - acc_dynamic),
            "p_value": p_value,
            "was_retrained": was_retrained * 1
        }

    return model_return, init_static, end_static, len_dynamic, was_retrained, values_graph
