from time import process_time_ns
from types import new_class
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
movies = pd.read_csv('movies.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv', low_memory=False)
asc = ratings.sort_values(by=['userId', 'timestamp'])
# print(asc)
# print(asc.values)

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
time_dataset = ratings.pivot(index='movieId',columns='userId',values='timestamp')
# print(time_dataset.head(10))
desc = ratings[ratings['userId'] == 1].sort_values(by='timestamp', ascending=False)
arr =desc.head(1).values
movie_list = arr[:,1]
movie_list = np.append(movie_list,16)
# print(movie_list)
# print(ratings)
# a= ratings.loc[movie_list]
# print(a)
# print(final_dataset)

final_dataset.fillna(0,inplace=True)
time_dataset.fillna(0,inplace=True)
# time_dataset.fillna(0,inplace=True)
# print(final_dataset.head())
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
# print(no_user_voted)
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# print(no_movies_voted[no_movies_voted < 50].index)

# f,ax = plt.subplots(1,1,figsize=(16,4))
# # ratings['rating'].plot(kind='hist')
# plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
# plt.axhline(y=10,color='r')
# plt.xlabel('MovieId')
# plt.ylabel('No. of users voted')
# plt.show()

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
# print(final_dataset)
#f,ax = plt.subplots(1,1,figsize=(16,4))
# plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
# plt.axhline(y=50,color='r')
# plt.xlabel('UserId')
# plt.ylabel('No. of votes by user')
# plt.show()

final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
# print(final_dataset)
final_dataset = final_dataset.loc[movie_list]
# print(final_dataset.T)
final_dataset = final_dataset.T
# b = cosine_similarity(final_dataset)
# np.fill_diagonal(b, 0 )
# print(b)
# similarity_with_user = pd.DataFrame(b,index=final_dataset.index)
# similarity_with_user.columns=final_dataset.index
# print(similarity_with_user.head())
# sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
# sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
# print(sparsity)
# csr_sample = csr_matrix(sample)
# print(csr_sample)
# print(final_dataset.head(10))
# csr_data = csr_matrix(final_dataset.values)
# print(final_dataset)
# print(csr_data)
# final_dataset.reset_index(inplace=True)
# print(final_dataset.head(10))

new_arr=  np.array(final_dataset)
list_data = new_arr.tolist()
# print(list_data)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(list_data)
distances , indices =knn.kneighbors([list_data[0]],n_neighbors=50)
rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
# print(rec_movie_indices)
list_uid = list(zip(*rec_movie_indices))[0]
print(list_uid)
array_phim = []
for i in list_uid:
        array_phim.append(ratings[ratings['userId'] == i].sort_values(by='timestamp', ascending=False))
# print(array_phim[0][array_phim[0]['movieId'] == 671].index)
for i in range(48):
        print(array_phim[i][array_phim[i]['movieId'] == 16].index)
# for i in 
# print(csr_data)
# print(new_arr)
# print(final_dataset[6])
# distances , indices = knn.kneighbors(final_dataset[4],n_neighbors=10+1)    

"""

"""
# def get_movie_recommendation(movie_name):
#     n_movies_to_reccomend = 10
#     movie_list = movies[movies['title'].str.contains(movie_name)]  
#     if len(movie_list):        
#         movie_idx= movie_list.iloc[0]['movieId']
#         movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
#         distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
#         rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
#         recommend_frame = []
#         for val in rec_movie_indices:
#             movie_idx = final_dataset.iloc[val[0]]['movieId']
#             idx = movies[movies['movieId'] == movie_idx].index
#             recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
#         df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
#         return df
#     else:
        # return "No movies found. Please check your input"
# print(csr_data)

# def get_final_dataset():
# movie_idx= movie_list.iloc[0]['movieId']
# print(movie_idx)
# print(final_dataset[final_dataset['userId'] == 11].index[0])
#  59315 1977
# print(final_dataset.head(10))
# print(final_dataset[final_dataset['movieId'] == 3].index[0])
# print(time_dataset.sort_values(by=['4']))
# def get_final_Data(uid,movie_name):
#     movie_list = movies[movies['title'].str.contains(movie_name)]  
#     movie_idx = movie_list.iloc[0]['movieId']
#     movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
#     # print(ratings[])
#     # print(ratings.at[])


# def get_movie_recommendation(movie_name,uid,rating):
#     n_movies_to_reccomend = 10
#     movie_list = movies[movies['title'].str.contains(movie_name)]  
#     if len(movie_list):        
#         # get index of movie
#         movie_idx= movie_list.iloc[0]['movieId']
#         movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]


#         distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
#         rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
#         recommend_frame = []
#         for val in rec_movie_indices:
#             movie_idx = final_dataset.iloc[val[0]]['movieId']
#             idx = movies[movies['movieId'] == movie_idx].index
#             recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
#         df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
#         return df
#     else:
#         return "No movies found. Please check your input"
# if __name__ == "__main__":
#     # print(get_movie_recommendation('Iron Man',1,1))
#     get_final_Data(1,"Iron Man")