import pandas as pd
import csv
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import pairwise_distances


def rcf(topk,n_similar):
    rs_cols = ['user_id', 'res_id', 'rating']  # train和test里面包含的列
    ratings_train = pd.read_csv('../../data/newtrain100.csv', names=rs_cols, index_col=False, header=None)
    ratings_test = pd.read_csv('../../data/newtest100.csv', names=rs_cols, index_col=False, header=None)

    n_users_train = len(ratings_train['user_id'].unique())  # 统计得到用户数目3291个
    n_items_train = len(ratings_train['res_id'].unique())

    n_users_test = len(ratings_test['user_id'].unique())  # 统计得到用户数目3291个
    n_items_test = len(ratings_test['res_id'].unique())

    # 统计得到用户数目3291个
    # 统计得到餐厅数目4088个 实际test里面还有一些train里面从未出现过的res_id
    # 一共4417个

    train_matrix = np.zeros((n_users_train, n_items_train))
    for line in ratings_train.itertuples():         # 存进train_matrix
        train_matrix[line[1], line[2]] = line[3]


    # n_similar=70
    # n_similar_user=10
    item_similarity = pairwise_distances(train_matrix.T, metric='cosine')#计算两个之间的距离用‘cosine'
    user_similarity = pairwise_distances(train_matrix,metric='cosine')

    similar_n = item_similarity.argsort()[:, -n_similar:][:, ::-1] #元素从小到大排列
    n_similar_user=10
    similar_n_user = user_similarity.argsort()[:, -n_similar_user:][:, ::-1]

    u, s, vt = svds(train_matrix, k=100)  # k是分解的纬度
    s_diag_matrix = np.diag(s)
    predictions_svd = np.dot(np.dot(u, s_diag_matrix), vt)

    # print('similar_n shape: ', similar_n_user.shape)
    # print(similar_n.shape)
    # print(similar_n_user)

    pred = np.zeros((n_users_train, n_items_train)) # 餐馆
    p1 = np.zeros((n_users_train, n_items_train)) #用户

    # print(pred.shape)

    for i, items in enumerate(similar_n): #i=3291
        similar_items_indexes = items
        similarity_n = item_similarity[i, similar_items_indexes]    #相似度
        matrix_n = train_matrix[:, similar_items_indexes]
        rated_items = matrix_n.dot(similarity_n) / similarity_n.sum() #平均
        pred[:, i] = rated_items # 存入pred

    # print(pred)

    for i, users in enumerate(similar_n_user):
        similar_users_indexes = users
        similar_n_user = user_similarity[i, similar_users_indexes]  # 相似度
        matrix_n = train_matrix[similar_users_indexes, :]
        rated_items = similar_n_user[:, np.newaxis].T.dot(
            matrix_n - matrix_n.mean(axis=1)[:, np.newaxis]) / similar_n_user.sum()
        p1[i, :] = rated_items

    # print(p1)


    testuserset = ratings_test['user_id'].unique()

    totalprecision = 0.0
    count = 0
    totalrecall=0.0
    totalf1=0.0
    for i in testuserset:
        userid = i # 是测试集合中的用户id
        # print(userid)
        user_ratings =   1*pred[userid, :]+ 9*predictions_svd[i, :]#取出pred餐馆 ，p1用户
        # print(user_ratings)
        train_unkown_indices = np.where(train_matrix[userid, :] == 0)[0] #判断该userid是否有相似度 ，若有则为1，否为0

        user_recommendations = user_ratings[train_unkown_indices] #把所有有相似度的userid导出
        recommendset = set()
        for res_id in user_recommendations.argsort()[-topk:][:: -1]: #整理得推荐列表{restid，restid。。。}
            recommendset.add(res_id)

        newtestset = set()
        testset = ratings_test[ratings_test['user_id'] == userid]['res_id'] # 判断测试集是否相等
        for i in testset:
            newtestset.add(i)  # newtestset->测试集

        # print(recommendset,newtestset)
        commonset = recommendset & newtestset
        precision = len(commonset) / topk
        recall=len(commonset)/len(newtestset)
        totalrecall+=recall
        totalprecision = totalprecision + precision
        if(precision+recall==0):
            f1=0
        else:
            f1=2*(precision*recall)/(precision+recall)
        totalf1=totalf1+f1
        count = count + 1

    # print()
    print(totalprecision / count)
    print(totalrecall/count)
    # print("f1")
    print(totalf1/count)


if __name__ == '__main__':
    # for i in range(1,6):
    for i in range(3,6):
        print(i)
        rcf(2,70)
