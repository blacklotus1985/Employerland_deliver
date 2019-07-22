# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:08:18 2018

@author: ggiannoni
"""


# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import configparser
import numpy as np
import os
from scipy.sparse.linalg import svds

# In[46]: data preparation



# In[2]: functions


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

def threshold_descriptions(matrix, threshold=0.5):
    """
    gets all the similarities for each description that are bigger of a certain threshold
    :param matrix: matrix of similarties
    :param threshold: fixed threshold
    :return: dictionary list of ID description-ID similarities
    """
    threshhold_list=[]
    for i in range(len(matrix[0])):
        cosine_desc = matrix[i]
        dict = {"job":i+1,"similar_ID":np.where(matrix[i]>threshold), "similarity_value":cosine_desc[np.where(cosine_desc>threshold)]}
        threshhold_list.append(dict)
    return threshhold_list


def get_recommendations(title, cosine_sim, threshold=0.035):
    # Get the index of the movie that matches the title

    idx = lookup_ch[lookup_ch['company_name'] == title].iloc[0, 2]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # scegli solo i valori sopra una certa soglia

    # sim_scores=
    sim_scores = list(filter(lambda x: x[1] > threshold, sim_scores))

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return lookup_ch['company_name'].iloc[movie_indices]


# initialize configurations and read data
conf = configparser.ConfigParser()
conf.read(os.path.dirname(os.getcwd())+'/configurations/configurations.ini')
challenge = pd.read_csv(os.path.dirname(os.getcwd())+conf.get("INPUT_FILES","challenge"),delimiter=";")


#clean data and aggregate data for company
ch_agg= challenge.groupby(['userID', 'companyID','name']).agg({'matchesDone': [sum] })

ch_agg.columns = ch_agg.columns.droplevel(level=0)

ch_agg['id_tab'] = ch_agg.index

ch_agg[['userID', 'companyID','company_name']] = ch_agg['id_tab'].apply(pd.Series)

ch_agg.drop(columns=['id_tab'])

ch_agg_mod_raw=ch_agg.drop(columns=['id_tab']).reset_index(drop=True)

ch_agg_mod=ch_agg_mod_raw.loc[:,['userID', 'companyID','sum']].sort_values(['companyID','userID'])

lookup_ch= ch_agg_mod_raw.loc[:,['companyID', 'company_name']].drop_duplicates().sort_values(['companyID']).reset_index(drop=True)

lookup_ch['index1'] = lookup_ch.index

n_users_ch = ch_agg.userID.unique().shape[0]
n_items_ch = ch_agg.companyID.unique().shape[0]

R_ch = ch_agg_mod.pivot(index = 'userID', columns ='companyID', values = 'sum').fillna(0)
ch_agg_mod.head()

 
R_matrix = R_ch.as_matrix()
user_ratings_mean_ch = np.mean(R_matrix, axis = 1)
R_demeaned_ch = R_matrix - user_ratings_mean_ch.reshape(-1, 1)


U, sigma, Vt = svds(R_demeaned_ch, k = 50)
sigma = np.diag(sigma)


vt_tran= Vt.T
cosine_sim = cosine_similarity(vt_tran, vt_tran)

description_index_list = top_desciptions(cosine_sim)
threshhold_list = threshold_descriptions(matrix=cosine_sim,threshold=0.5)
indices = pd.Series(lookup_ch.index, index=lookup_ch['company_name']).drop_duplicates()

# In[46]: test
idx = lookup_ch[lookup_ch['company_name']=='Antica Focacceria San Francesco'].iloc[0,2]
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:11]


sim_scores = list(filter(lambda x: x[1] > 0.04, sim_scores))

movie_indices = [i[0] for i in sim_scores]


output = pd.DataFrame(columns=['comp1','comp2'])

for comp in lookup_ch['company_name']:
    related_company=get_recommendations(comp,cosine_sim=cosine_sim)
    for x in related_company:
        output = output.append({'comp1': comp,'comp2' :x  }, ignore_index=True)

output.to_csv(os.path.dirname(os.getcwd())+conf.get("OUTPUT_FILES","folder")+"items_companies.csv",sep=";",index=None)

