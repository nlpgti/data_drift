import re
from multiprocessing import Pool
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def run_in_parallel(experiment_run, list_to_pool):
    p = Pool()
    list_data = p.map(experiment_run, list_to_pool)
    p.close()
    p.join()
    return list_data


def get_stopwords():
    english_stopwords = [elem for elem in stopwords.words('english') if
                         not (elem in ['no', 'not', 'nor', 'yes', 'up', 'down', 'under', 'few', 'more', 'most'])]
    english_stopwords.append("Â")
    list_stopwords = list(set(english_stopwords))
    return list_stopwords


def get_re_expression(list_words):
    if len(list_words) == 0:
        return None
    list_words.sort(reverse=True)
    big_regex_special_names = r'|'.join(map(re.escape, list_words))
    return r"\b(" + big_regex_special_names + r")\b"


def get_char_re_expression(list_words):
    if len(list_words) == 0:
        return None
    list_words.sort(reverse=True)
    big_regex_special_names = r'|'.join(map(re.escape, list_words))
    return r"(" + big_regex_special_names + r")"


def print_experiment_metrics(model_name, data, y, y_pred, time):
    accuracy_value = accuracy_score(y, y_pred)
    f1_score_values = f1_score(y, y_pred, average=None, labels=["nofake", "fake"])

    str_result = "data: " + str(data) + "\n" + str(
        confusion_matrix(y, y_pred, labels=["nofake", "fake"])) + "\n" + "y_len: " + str(
        len(y)) + "\n" + "y_pred: " + str(len(y_pred)) + "\n" + "Accuracy: " + str(
        accuracy_score(y, y_pred)) + "\n" + "Precision: " \
                 + str(precision_score(y, y_pred, average=None, labels=["nofake", "fake"])) + "\n" + "Recall: " \
                 + str(recall_score(y, y_pred, average=None, labels=["nofake", "fake"])) + "\n" + "F1: " \
                 + str(f1_score(y, y_pred, average=None, labels=["nofake", "fake"])) + "\n"

    recall = recall_score(y, y_pred, average=None, labels=["nofake", "fake"])

    latex_result = "\\textsc{" + model_name + "}" + " & {:.2f}".format(accuracy_value * 100) \
                   + " & {:.2f}".format(f1_score_values.mean() * 100) \
                   + " & {:.2f}".format(f1_score_values[0] * 100) \
                   + " & {:.2f}".format(f1_score_values[1] * 100) \
                   + " & {:.2f}".format(time) \
                   + "\\\\"

    return {"data": data, "text": str_result, "latex": latex_result, "recall_fake": recall[1], "acc": accuracy_value,
            "f1_score_fake": f1_score_values[1]}
