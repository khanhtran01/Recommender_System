from typing import List
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

MIN_USER_VOTED = 3 # so voted 1 phim it nhat phai co
MIN_MOVIE_VOTED = 50 # so phim it nhat 1 user phai vote

np.set_printoptions(suppress=True)
# doc file
movies = pd.read_csv('movies.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv', low_memory=False)
# asc = file rating sort theo userId va time
asc = ratings.sort_values(by=['userId', 'timestamp'])
# đếm số lượt được voted của film
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
# đếm số lần vote của 1 user
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
# print(no_user_voted)

# lọc ra những user và movie không thỏa điều kiện
ratings_temp = ratings[ratings.movieId.isin(no_user_voted[no_user_voted> MIN_MOVIE_VOTED].index)]
ratings_temp = ratings_temp[ratings_temp.userId.isin(no_movies_voted[no_movies_voted>MIN_USER_VOTED].index)]
print(ratings_temp)
print(ratings)
def getListUid(uid):
    """Trả về danh sách userId giống với uid"""
    ratings_temp = ratings    
    # chuyển dữ liệu thành bảng 2x2 với row = movideId, column = userId và value = rating
    final_dataset = ratings_temp.pivot(index='movieId',columns='userId',values='rating')
    # set các giá trị N/A thành giá trị trung bình của column tương ứng
    # final_dataset = final_dataset.fillna(final_dataset.mean(axis=0))

    #sort file rating theo userId và time
    desc = ratings[ratings['userId'] == uid].sort_values(by='timestamp', ascending=False)
    # lấy 5 giá trị có time lớn nhất cho user đã chọn
    try:
        arr = desc.head(5).values
        
    except NameError:
        print("Nguoi dung phai xem it nhat 5 phim")

    # chuyển sang dạng list và tách chỉ lấy list movie
    movie_list = arr[:,1]
    
    
    # trên dữ liệu final_dataset chỉ lấy những bộ film có trong list ở trên
    final_dataset = final_dataset.loc[movie_list]
    #set các giá trị N/A thành 0
    final_dataset.fillna(0,inplace=True)
    # xoay bảng để row = userId và clolumn = movieId
    # phải xoay theo định dạng của hàm knn sẽ sử dụng
    final_dataset = final_dataset.T
    new_arr=  np.array(final_dataset)
    list_data = new_arr.tolist()
    # khai báo KNN và set giá giá trị thuật toán cho nó
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    # chọn dataset cho thuật toán
    knn.fit(list_data)
    # trả về 10 user gần giống với uid nhất
    distances , indices =knn.kneighbors([list_data[0]],n_neighbors=10)
    return sorted(zip(distances.tolist()[0], indices.tolist()[0]))

def checksize (uid):
    desc = ratings[ratings['userId'] == uid].sort_values(by='timestamp', ascending=False)
    size = desc.size/4
    if size > MIN_MOVIE_VOTED:
        return True
    return False


def get_recommendation(uid):
    # kiểm tra uid có thuộc diện bị lọc bnỏ hay k
    # if (checksize(uid) == False):
    #     return extend_mode(uid)
    list_uid = getListUid(uid)

    list_uid.pop(0)
    ratings_temp = ratings
    # test_id = ratings_temp[ratings_temp['userId'] == uid]

    # x = int(test_id.size/4)
    count_phim = []
    np.set_printoptions(suppress=True,\
    formatter={'float_kind':'{:0.0f}'.format})


    # lọc qua từng user và đếm số vote >4 của từng bộ phim
    for i in list_uid:
        test_id = ratings_temp[ratings_temp['userId'] == i[1]]
        x = int(test_id.size/4)
        for j in range(x):
            if (test_id.iloc[j].at['rating'] >= 4):
                count_phim.append(test_id.iloc[j].at['movieId'])
    unique, counts = np.unique(np.array(count_phim), return_counts=True)
    final_arr = np.asarray((unique, counts)).T.tolist()
    # sort các bộ phim theo số lần được vote >= 4 nhiều nhấ
    top_recommendation = sorted(final_arr,key=lambda l:l[1], reverse=True)
    list_phim = []
    # lấy được 10 bộ phim
    for i in range(10):
        list_phim.append(top_recommendation[i][0])
    list_movie_name = []
    for i in list_phim:
        list_movie_name.append({'Title':movies.loc[i]['title']})
    # recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
    df = pd.DataFrame(list_movie_name,index=range(1,10+1))
    return df

if __name__ == "__main__":
    while(True):
        uid = int(input("Nhap user ID: "))
        if (uid == 0):
            break
        print(get_recommendation(uid))