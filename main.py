from typing import List
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

MIN_USER_VOTED = 3 # so voted 1 phim it nhat phai co
MIN_MOVIE_VOTED = 50 # so phim it nhat 1 user phai vote

np.set_printoptions(suppress=True)
movies = pd.read_csv('movies.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv', low_memory=False)
asc = ratings.sort_values(by=['userId', 'timestamp'])
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
# print(no_user_voted[no_user_voted>3].index)
# print(ratings)
# print(no_movies_voted[no_movies_voted>MIN_USER_VOTED])
# new_rating = ratings.loc[ratings['userId'] == no_user_voted[no_user_voted>3].index]
ratings = ratings[ratings.movieId.isin(no_user_voted[no_user_voted> MIN_MOVIE_VOTED].index)]
ratings = ratings[ratings.userId.isin(no_movies_voted[no_movies_voted>MIN_USER_VOTED].index)]
def getListUid(uid):
    ratings_temp = ratings    
    final_dataset = ratings_temp.pivot(index='movieId',columns='userId',values='rating')
    final_dataset = final_dataset.fillna(final_dataset.mean(axis=0))
    desc = ratings_temp[ratings_temp['userId'] == uid].sort_values(by='timestamp', ascending=False)
    arr = desc.head(5).values
    movie_list = arr[:,1]
    final_dataset = final_dataset.loc[movie_list]
    final_dataset.fillna(0,inplace=True)

    final_dataset = final_dataset.T
    new_arr=  np.array(final_dataset)
    list_data = new_arr.tolist()
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    knn.fit(list_data)
    distances , indices =knn.kneighbors([list_data[0]],n_neighbors=10)
    return sorted(zip(distances.tolist()[0], indices.tolist()[0]))

def checksize (uid):
    desc = ratings[ratings['userId'] == uid].sort_values(by='timestamp', ascending=False)
    size = desc.size/4
    if size > MIN_MOVIE_VOTED:
        return True
    return False

def extend_mode(uid):
    final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
    final_dataset.fillna(0,inplace=True)
    no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

    final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
    final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)

    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)

    # def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    # movie_list = movies[movies['title'].str.contains(movie_name)]  
    desc = ratings[ratings['userId'] == uid].sort_values(by='timestamp', ascending=False)
    arr = desc.head(1).values
    movie_list = arr[:,1]

          
    movie_idx= movie_list[0]
    movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
    distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
    rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
    recommend_frame = []
    for val in rec_movie_indices:
        movie_idx = final_dataset.iloc[val[0]]['movieId']
        idx = movies[movies['movieId'] == movie_idx].index
        recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
    df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1)).sort_values(by='Distance', ascending=True)
    return df

def get_recommendation(uid):
    if (checksize(uid) == False):
        return extend_mode(uid)
    list_uid = getListUid(uid)
    # print(list_uid)
    list_uid.pop(0)
    ratings_temp = ratings
    test_id = ratings_temp[ratings_temp['userId'] == uid]

    x = int(test_id.size/4)
    count_phim = []
    np.set_printoptions(suppress=True,\
    formatter={'float_kind':'{:0.0f}'.format})
    for i in list_uid:
        test_id = ratings_temp[ratings_temp['userId'] == i[1]]
        x = int(test_id.size/4)
        for j in range(x):
            if (test_id.iloc[j].at['rating'] >= 4):
                count_phim.append(test_id.iloc[j].at['movieId'])
    unique, counts = np.unique(np.array(count_phim), return_counts=True)
    final_arr = np.asarray((unique, counts)).T.tolist()
    top_recommendation = sorted(final_arr,key=lambda l:l[1], reverse=True)
    list_phim = []
    for i in range(10):
        list_phim.append(top_recommendation[i][0])
    list_movie_name = []
    for i in list_phim:
        list_movie_name.append({'Title':movies.loc[i]['title']})
    # recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
    df = pd.DataFrame(list_movie_name,index=range(1,10+1))
    return df

# print(get_recommendation(2))
if __name__ == "__main__":
    while(True):
        uid = int(input("Nhap user ID: "))
        if (uid == 0):
            break
        print(get_recommendation(uid))