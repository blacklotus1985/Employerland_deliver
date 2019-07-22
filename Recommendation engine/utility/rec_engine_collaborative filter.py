
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import configparser
from nltk.corpus import stopwords
from nltk.stem.snowball import ItalianStemmer
import numpy as np
import os
import re

# read input files, create configuration instance and make a copy of df
conf = configparser.ConfigParser()
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
conf.read(os.path.dirname(os.getcwd())+'/configurations/configurations.ini')
filename = conf.get("INPUT_FILES",conf.get("INPUT_FILES","input"))
df = pd.read_csv(os.path.dirname(os.getcwd())+filename,delimiter=';' )
old_df = df.copy()


def clean_stop_words(df,column,lang,stem=True):
    """
    (df,str,str) -> df
    cleans column of dataframe from stopwords with the given language
    :param df: dataframe to clean
    :param column: column of dataframe to clean
    :param lang: language of stopwords
    :param stem: if stemming activated
    :return: cleaned dataframe
    """
    for i in range(df.shape[1]):
        df.loc[i,column]=re.sub('[^a-zA-Z]', ' ', df[column][i])
    document = df[column].str.lower().str.split()
    sentence_stem = []
    document_stem = []

    nltk_stop = stopwords.words(lang)
    clean_document = document.apply(lambda x: [item for item in x if item not in nltk_stop])
    stemmer = ItalianStemmer()
    if stem:
        for sentence in clean_document:
            for word in sentence:
                word = stemmer.stem(word)
                sentence_stem.append(word)
            document_stem.append(sentence_stem)
            sentence_stem = []
        sentences = [' '.join(i) for i in document_stem]
        cleaned_series = pd.Series((v for v in sentences))
        df[column] = cleaned_series
    else:
        sentences = [' '.join(i) for i in clean_document]
        cleaned_series = pd.Series((v for v in sentences))
        df[column] = cleaned_series
    return df


def n_top_items(x, n_items, max=True):
    """
    get index of max n_items from array
    :param x: array
    :param n_items: number of items
    :param max: if True gives indexes of max, if False gives indexes of min
    :return: indexes of array
    """
    if max:
        sort_array = x.argsort()[-n_items:][::-1]
    else:
        sort_array = x.argsort()[:n_items][::1]
    return sort_array

def find_best_words(df,matrix,word_dict,n,conf,filename="default",save=True):
    """
    retrieves most significant words from tfidf matrix
    :param matrix: sparse matrix
    :param word_dict: dictionary of index and words of tfidf
    :param n: number of words retrieved
    :return: dataframe with job id -  top words
    """

    list=[]
    final_terms=[]
    final_dict_list=[]
    for i in range(matrix.shape[0]):
        arr = matrix[i].toarray()[0]
        top_items = n_top_items(arr,n_items=n,max=True)
        list.append(top_items)
        for elem in list[0]:
            final_terms.append(word_dict[elem])
        dict={"job_ID":df["ID"][i], "top_words":final_terms}
        final_dict_list.append(dict)
        list=[]
        final_terms=[]
    data_frame_id_words = pd.DataFrame.from_records(final_dict_list,coerce_float=True)

    if save:
        wd = os.getcwd()
        os.chdir(wd)
        data_frame_id_words.to_csv(os.path.dirname(os.getcwd())+conf.get("OUTPUT_FILES","folder") + filename, sep=";", index=False)
    return final_dict_list, data_frame_id_words


def top_desciptions(matrix, n=5):
    """
    gets the top similarities for each description
    :param matrix: matrix of similarity
    :param n: number of top similarities
    :return: list of similarities
    """
    description_index_list =[]
    for i in range(len(matrix[0])):
        array_res = n_top_items(matrix[i], n_items=n, max=True)
        description_index_list.append(array_res)
    return description_index_list

def threshold_descriptions(df,matrix, conf, threshold=0.5,filename="default",save=True):
    """
    gets all the similarities for each description that are bigger of a certain threshold
    :param matrix: matrix of similarties
    :param threshold: fixed threshold
    :return: data frame of dictionary list of Job ID - similarityID - Similarity value
    """
    threshhold_list=[]
    for i in range(len(matrix[0])):
        cosine_desc = matrix[i]
        dict = {"job_ID":df["ID"][i],"similar_ID":df["ID"][np.where(matrix[i]>threshold)[0]].values, "similarity_value":np.asarray(cosine_desc[np.where(cosine_desc>threshold)[0]])}
        threshhold_list.append(dict)
    df_threshold = pd.DataFrame.from_records(threshhold_list,coerce_float=True)

    if save:
        wd = os.getcwd()
        os.chdir(wd)
        df_threshold.to_csv(os.path.dirname(os.getcwd())+conf.get("OUTPUT_FILES","folder")+filename,sep=";", index = False)
    return threshhold_list,df_threshold

def get_recommendations(title, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]

# clean descriptions from italian stopwords
df = clean_stop_words(df=df, column="description", lang = "italian",stem=True)

# clean descriptions
df['description'] = df['description'].fillna('')

# create tfidf model instance
tfidf = TfidfVectorizer(stop_words='english')

# apply tfidf model to description column and create tf idf matrix
tfidf_matrix = tfidf.fit_transform(df['description'])

# swamp key value of vocabulary name-index
word_indexes = tfidf.get_feature_names()
dict_vocab = tfidf.vocabulary_
swap_vocab = {v:k for k,v in dict_vocab.items()}


# calculate cosine similarity for the embedded vectors of the job positions
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# find the most 5 representative words for each job position and save it into csv file
final_dict_list, data_frame_id_words = find_best_words(df=df,matrix=tfidf_matrix,conf=conf,word_dict=swap_vocab,n=5,filename="id_words.csv")

# creates a list for each description and the index of the best 5 descriptions
description_index_list = top_desciptions(cosine_sim)

# loops all the description and gets indexes of all the descriptions that are within a threshold of similarity.
threshhold_list,df_threshold = threshold_descriptions(df=df,matrix=cosine_sim,conf=conf,threshold=0.3,filename="threshold_descriptions.csv")


# drop duplicates from column
indices = pd.Series(df.index, index=df['title']).drop_duplicates()







# get recommendations for given title
indices_final = get_recommendations('Neolaureati in Ingegneria Informatica/Informatica',cosine_sim=cosine_sim)




