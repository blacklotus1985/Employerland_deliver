
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:12:26 2018

@author: ggiannoni,azeisberg
"""

# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import configparser
import os
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import scale



#read configuration file
conf = configparser.ConfigParser()
conf.read(os.path.dirname(os.getcwd())+r'/configurations/configurations.ini')



def find_rated_challenges_indexes(rating_matrix):
    """
    find indexes of all challenges that have been rated
    :param rating_matrix: matrix of challenges taken by user
    :return: list of list of indexes of done challenges
    """
    rating_list=[]
    for i in range(rating_matrix.shape[0]):
        rating_list.append(np.where(rating_matrix[i,:]!=0)[0])
    return rating_list

def count_bad_user(rating_list, id_company=6):
    """
    finds how many and which users played at most one challenge for the company
    :param rating_list: list of challenges taken by users
    :param id_company: id of the company to analyze
    :return: count of bad users and list of their indexes
    """
    i=0
    j=0
    index_list=[]
    for elem in rating_list:
        if len(elem) ==1 and elem[0]==id_company:
            i = i+1
            index_list.append(j)
        j=j+1
    return i,index_list

def clean_matrix(matrix,list_to_delete):
    """
    cleans matrix from bad users
    :param matrix: matrix to clean
    :param list_to_delete: list of users to delete
    :return: cleaned matrix
    """
    matrix = np.delete(matrix,(list_to_delete),axis=0)
    return matrix

def random_whitening(rating_list_cleaned, probability=0.2):
    """
    select random challenge with p probability among all challenges done
    :param rating_list_cleaned: rating list cleaned from test app users
    :param probability: probability of whitening a cell
    :return: list of random selected challenges
    """
    random_list =[]
    for elem in rating_list_cleaned:
        random_values_per_user = np.random.uniform(size=len(elem))
        index_selection = np.where(random_values_per_user>probability)
        random_list.append(elem[index_selection])
    return random_list

def find_whitened_cells(random_list_user_whiten, rating_list_cleaned):
    """
    finds whitened cells for every user
    :param random_list_user_whiten:
    :param rating_list_cleaned:
    :return: list of whitened cells list
    """

    whitened_cells_list=[]
    for i in range(len(rating_list_cleaned)):
        white_cells_per_user = list(set(random_list_user_whiten[i]).symmetric_difference(rating_list_cleaned[i]))
        whitened_cells_list.append(white_cells_per_user)
    return whitened_cells_list

def create_test_dataframe(whitened_cells_list,R_matrix):
    """
    create new dataframe putting 0 value to random cells
    :param whitened_cells_list: random cells
    :param R_matrix: Matrix of preferences
    :return: Matrix with whitened cells
    """
    R_matrix_copy = R_matrix.copy()
    for i in range(R_matrix.shape[0]):
        for elem in whitened_cells_list[i]:
            R_matrix_copy[i][elem]=0
    return R_matrix_copy


def recommendation_calc(matrix, k =30, single_user_id=1,top_preferences=5):
    """
    calculates predicted challenges for users and single users preferences
    :param matrix: matrix to analyze
    :param k: dimension of users and item matrix
    :param single_user_id: user to obtain preferences
    :param top_preferences: number of preferences per user
    :return: prefernces of single user and all users
    """
    U, sigma, Vt = svds(matrix, k = k)
    sigma_diagonal = np.diag(sigma)
    all_user_predicted_ratings_ch = np.dot(np.dot(U, sigma_diagonal), Vt)
    preds_df_ch = pd.DataFrame(all_user_predicted_ratings_ch, columns = R_ch.columns)
    user_full, recommendations = item_recommendation(preds_df_ch, single_user_id, lookup_ch, ch_agg_mod, top_preferences)

    return user_full,recommendations,all_user_predicted_ratings_ch,preds_df_ch


def values_matrixes(matrix_pref, matrix_white, whitened_cells_list):
    """
    list of all predictions of cells that were whitened used for testing
    :param matrix_pref: matrix of preferences
    :param matrix_white: matrix of preferences whitened
    :param whitened_cells_list: list of whitened cells
    :return: arrays of predictions and rmse
    """
    matrix_pref_list = []
    matrix_whitened_list = []
    for i in range(len(whitened_cells_list)):
        for elem in whitened_cells_list[i]:
            matrix_pref_list.append(matrix_pref[i,elem])
            matrix_whitened_list.append(matrix_white[i,elem])
    matr_pref_array = np.asarray(matrix_pref_list)
    matr_whitened_array = np.asarray(matrix_whitened_list)
    rmse = np.sqrt(mean_squared_error(matr_pref_array,matr_whitened_array))

    return matrix_whitened_list, matrix_pref_list,rmse

def find_good_users_id(user_id_list, index_bad_list):
    """
    given list of bad users returns user_id of good users
    :param user_id_list: user_id complete list
    :param index_bad_list: index of bad users
    :return: list of good users
    """

    discarded_users = user_id_list[index_bad_list]
    good_users_id_set = set(user_id_list)-set(discarded_users)
    good_users_id_list = list(good_users_id_set)
    good_users_id_list.sort()
    return good_users_id_list

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
def save_df(df,conf,filename="default.csv",index=False,sep=";"):
    """
    save dataframe into folder from path
    :param df: dataframe to save
    :param conf: configuration file
    :param filename: filename to save
    :param index: if True creates row indexes
    :param sep: separator of file
    :return: None
    """
    base_path = os.path.dirname(os.getcwd())
    folder = conf.get("OUTPUT_FILES","folder")
    df.to_csv(base_path+folder+filename, sep=sep,index=index)

def item_recommendation(predictions_df, userID, item_df, original_ratings_df, num_recommendations=5):

    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the challenges information.
    user_data = original_ratings_df[original_ratings_df.userID == (userID)]
    user_data.reset_index(drop=True,inplace=True)
    user_full = (user_data.merge(item_df, how ='left', left_on ='companyID', right_on ='companyID').
                     sort_values(['sum'], ascending=False))

    # Recommend the highest predicted number of challenges that the user hasn't seen yet.
    recommendations = (item_df[~item_df['companyID'].isin(user_full['companyID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'companyID',
               right_on = 'companyID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# pre processing
filename = conf.get("INPUT_FILES",conf.get("INPUT_FILES","input"))
challenge = pd.read_csv(os.path.dirname(os.getcwd()) + conf.get("INPUT_FILES", "challenge"), delimiter=';')
lookup_ch= challenge.loc[:, ['companyID', 'name']].drop_duplicates()
ch_agg= challenge.groupby(['userID', 'companyID']).agg({'matchesDone': [sum]})
ch_agg.columns = ch_agg.columns.droplevel(level=0)
ch_agg['index1'] = ch_agg.index
ch_agg[['userID', 'companyID']] = ch_agg['index1'].apply(pd.Series)
ch_agg_mod=ch_agg.drop(columns=['index1'])


# list of unique ID of users
user_id_list = ch_agg.userID.unique()

# list of unique ID of companies
company_id_array = ch_agg.companyID.unique()

# create pivot table
R_ch = ch_agg_mod.pivot(index = 'userID', columns ='companyID', values = 'sum').fillna(0)
corr_list = []


# R_matrix is the raw matrix of the challenges made per user
R_matrix = R_ch.values

# find list of movies rated by user
rating_list = find_rated_challenges_indexes(R_matrix)


# find the users that played only test challenge
counter,index_list = count_bad_user(rating_list=rating_list)

# cleaned matrix from users that played only 1 challenge
cleaned_matrix = clean_matrix(matrix=R_matrix,list_to_delete=index_list)

# create copy of cleaned matrix
R_matrix = cleaned_matrix.copy()

# indexes of challenges made by users cleaned
rating_list_cleaned = find_rated_challenges_indexes(R_matrix)

# indexes of challenges whitened with p probability
random_list_user_whiten = random_whitening(rating_list_cleaned,probability=0.2)


# find cells that will be whitened from dataframe
whitened_cells_list = find_whitened_cells(random_list_user_whiten=random_list_user_whiten,rating_list_cleaned=rating_list_cleaned)


# whitened matrix of user preferences
R_matrix_whitened = create_test_dataframe(whitened_cells_list=whitened_cells_list,R_matrix=R_matrix)

# normalize sparse matrix data
scale(R_matrix,copy=False)
scale(R_matrix_whitened,copy=False)
# calculates users that are good for analysis
good_users_id_list = find_good_users_id(user_id_list=user_id_list,index_bad_list=index_list)
dict_list =[]

for elem in good_users_id_list:
    try:
        user_full, recommendations, all_user_predicted_ratings_ch, preds_df_ch = recommendation_calc(matrix=R_matrix, k=conf.getint("ITEMS","svd_dimension"),single_user_id=elem,top_preferences=conf.getint("ITEMS","top_preferences"))
        user_full_white,recommendations_white,all_user_predicted_ratings_ch_white,preds_df_ch_white= recommendation_calc(matrix=R_matrix_whitened,k=conf.getint("ITEMS","svd_dimension"),single_user_id=elem,top_preferences=conf.getint("ITEMS","top_preferences"))

        # RMSE of test and values of cells of two predicted matrices
        matrix_whitened_list, matrix_pref_list,rmse=values_matrixes(matrix_pref=all_user_predicted_ratings_ch, matrix_white=all_user_predicted_ratings_ch_white, whitened_cells_list=whitened_cells_list)
        top_ratings = n_top_items(all_user_predicted_ratings_ch[elem],n_items=conf.getint("ITEMS","top_preferences"),max=True)
        dict={}
        dict = {"User_Id":np.unique(user_full["userID"].values)[0], "rated_companies":user_full["name"].values, "suggested_companies":recommendations["name"].values}
        if "User_Id" in dict:
            dict_list.append(dict)
    except:
        pass


# create pandas dataframe with list of results
data_frame_results = pd.DataFrame(dict_list)

# save pandas dataframe to csv
save_df(df=data_frame_results,conf=conf,filename="user_preferences.csv")





















