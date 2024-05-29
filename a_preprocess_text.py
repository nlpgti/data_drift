import datetime
import os
import warnings
import re
from utils.constants import CharsToRemove
from multiprocessing import Pool
import pandas as pd
import spacy
from utils.utils import get_re_expression, get_stopwords
import textdescriptives as td
import textstat
import text2emotion as te
from langdetect import DetectorFactory
from spacytextblob.spacytextblob import SpacyTextBlob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
DetectorFactory.seed = 0

re_stopwords = get_re_expression(get_stopwords())
nlp = spacy.load('en_core_web_md', disable=["ner", "senter"])
nlp.add_pipe('textdescriptives')
nlp.add_pipe('spacytextblob')


def get_number_urls(text):
    urls = []
    matches = re.finditer(
        r"(?i)\b((?:https?:\/|pic\.|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
        text, re.MULTILINE | re.IGNORECASE)
    for matchNum, match in enumerate(matches, start=1):
        urls.append(match.group())
    return len(urls)


def lemmatization_text(text):
    original_text = text
    result = {}
    doc = nlp(text)
    text_lemmatizer = ' '.join(e.lemma_ for e in doc)
    text_lemmatizer = re.sub(re_stopwords, " ", text_lemmatizer, flags=re.IGNORECASE)
    text_lemmatizer = re.sub(r"\s+", " ", text_lemmatizer)

    result["pos_prop_ADJ"] = td.extract_df(doc).iloc[0].get("pos_prop_ADJ")
    result["pos_prop_ADV"] = td.extract_df(doc).iloc[0].get("pos_prop_ADV")
    result["char_counts"] = len(original_text)

    out_emotion = te.get_emotion(original_text)
    result["Angry"] = out_emotion["Angry"]
    result["Fear"] = out_emotion["Fear"]
    result["Happy"] = out_emotion["Happy"]
    result["Sad"] = out_emotion["Sad"]
    result["Surprise"] = out_emotion["Surprise"]
    result["flesch_reading_ease"] = textstat.flesch_reading_ease(original_text)
    result["pos_prop_INTJ"] = td.extract_df(doc).iloc[0].get("pos_prop_INTJ")
    result["mcalpine_eflaw"] = textstat.mcalpine_eflaw(original_text)
    result["polarity"] = doc._.polarity
    result["pos_prop_PUNCT"] = td.extract_df(doc).iloc[0].get("pos_prop_PUNCT")
    result["reading_time"] = textstat.reading_time(original_text)
    result["pos_prop_VERB"] = td.extract_df(doc).iloc[0].get("pos_prop_VERB")
    result["urls"] = get_number_urls(original_text)
    result["pos_prop_NOUN"] = td.extract_df(doc).iloc[0].get("pos_prop_NOUN")
    result["pos_prop_PRON"] = td.extract_df(doc).iloc[0].get("pos_prop_PRON")
    result["difficult_words_count"] = textstat.difficult_words(original_text)
    result["text_preprocessed"] = text_lemmatizer

    return result


def preprocess_text(text):
    text = re.sub(
        r"(?i)\b((?:https?:\/|pic\.|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
        "", text)
    text = re.sub(r"\b[\w|\.||=-]+@[\w|\.|-]+\b", "", text)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
    text = re.sub(r"(?:(pic.|http|www|\w+)?\:(//)*)\S+", " ", text)

    result = lemmatization_text(text)
    text = result["text_preprocessed"]

    text = re.sub(r"(\*|\[|\]|=|\(|\)|\$|\"|\}|\{|\||\+|&|€|£|/|º)+", " ", text)
    text = re.sub(r"(\s|\t|\\n|\n)+", " ", text)
    text = re.sub(r"[\,|\.|'|:|;|\-|–]+", " ", text)
    text = re.sub(r"\d+[A-Za-z]*", " ", text)

    text = re.sub(CharsToRemove.re_list_chars_to_remove, " ", text, flags=re.IGNORECASE)

    list_words = re.findall(r"\b[a-zA-Z\-]{2,}\b", text)
    text = " ".join(list_words)
    text = re.sub(r"\s+", " ", text)

    text = text.lower()
    result["text_preprocessed"] = text

    return result


def lemmatization_clean_text(n_samples, column_tolemmatize, completed_path):
    dataset = pd.read_csv(completed_path, sep=",", engine="pyarrow")[:n_samples]
    p = Pool()
    list_data = p.map(preprocess_text, dataset[column_tolemmatize])
    p.close()
    p.join()
    array_result = pd.DataFrame(list_data)

    dataset = pd.concat([dataset, array_result], axis=1)

    return_path = "datasets/text_preprocess_dataset.csv"
    dataset.to_csv(return_path, index=False, header=True)
    return return_path


def get_year(date):
    return date.split("/")[2]


def date_to_timestamp(data_str, format_str):
    return datetime.datetime.strptime(data_str, format_str).timestamp()


def get_number_words(text):
    return len(text.split())


def target_num_to_category(x):
    if x == -1:
        return "fake"
    elif x == 1:
        return "nofake"


def prepare_dataset(path):
    dataset = pd.read_csv(path, engine="pyarrow")
    dataset["Timestamp"] = dataset["Date"].apply(
        lambda x: date_to_timestamp(x, '%m/%d/%Y'))
    dataset["Year"] = dataset["Date"].apply(
        lambda x: get_year(x))
    dataset["Word_count"] = dataset["Review"].apply(lambda x: get_number_words(x))
    dataset["Label"] = dataset["Label"].apply(lambda x: target_num_to_category(x))

    dataset = dataset.sort_values(by=['Timestamp'], ascending=True)
    dataset = dataset.reset_index(drop=True)
    dataset.to_csv("datasets/preprocessed_dataset.csv", index=False, header=True)
    return lemmatization_clean_text(None, "Review", "datasets/preprocessed_dataset.csv")


if __name__ == '__main__':
    prepare_dataset("datasets/Labelled_Yelp_Dataset.csv")
