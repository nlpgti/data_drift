import itertools


def hyperparameter_hatc(data):
    max_depth = data["max_depth"]
    tie_threshold = data["tie_threshold"]
    max_size = data["max_size"]
    remove_poor_attrs = data["remove_poor_attrs"]
    all_elements_to_evaluate = [max_depth, tie_threshold, max_size, remove_poor_attrs]
    hyper_parameters_list = list(itertools.product(*all_elements_to_evaluate))
    return hyper_parameters_list


def hyperparameters_arfc(data):
    n_models = data["n_models"]
    max_features = data["max_features"]
    lambda_value = data["lambda_value"]
    all_elements_to_evaluate = [n_models, max_features, lambda_value]
    hyper_parameters_list = list(itertools.product(*all_elements_to_evaluate))
    return hyper_parameters_list
