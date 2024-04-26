import csv
import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import pairwise_distances


def sop(topk):

    rs_cols = ['user_id', 'res_id', 'rating']  # train和test里面包含的列
    ratings_train = pd.read_csv('../../data/newtrain100.csv', names=rs_cols, index_col=False, header=None)
    ratings_test = pd.read_csv('../../data/newtest100.csv', names=rs_cols, index_col=False, header=None)

    n_users_train = len(ratings_train['user_id'].unique())  # 统计得到用户数目3291个
    n_items_train = len(ratings_train['res_id'].unique())

    popdict=dict() #统计餐馆res_id出现次数的字典
    with open('../../data/newtrain100.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            userid = data[0]
            res_id=data[1]
            rating=data[2]
            if(popdict.__contains__(res_id)): #res_id是否在字典中
                popdict[res_id]=popdict[res_id]+1 #在，则统计加一
            else:
                popdict.__setitem__(res_id,1) #不,在就增加字典res_id并赋值1


    # print(popdict)

    #作归一化，使得值在0-1之间

    maxvalue = max(popdict.values())
    newpopdict=dict()#归一化后的统计数组
    for k,v in popdict.items():
        newpopdict.__setitem__(k,v/maxvalue)

    newpop = sorted(newpopdict.items(), key=lambda x: x[1], reverse=True)#根据第二列的字段进行排序
    # reverse=True是从大到小排列


    # print(newpopdict)

    train_matrix = np.zeros((n_users_train, n_items_train))

    for line in ratings_train.itertuples():
        train_matrix[line[1], line[2]] = line[3]

    n_similar = 50
    item_similarity = pairwise_distances(train_matrix.T, metric='cosine')
    similar_n = item_similarity.argsort()[:, -n_similar:][:, ::-1]
    # print('similar_n shape: ', similar_n.shape)
    pred = np.zeros((n_users_train, n_items_train))
    pred = np.zeros((n_users_train, n_items_train))
    for i, items in enumerate(similar_n):
        similar_items_indexes = items
        similarity_n = item_similarity[i, similar_items_indexes]
        matrix_n = train_matrix[:, similar_items_indexes]
        rated_items = matrix_n.dot(similarity_n) / similarity_n.sum()
        pred[:, i] = rated_items

    # print(predictions_svd.shape)
    # print(predictions_svd)

    testuserset = ratings_test['user_id'].unique()
    testrestset = ratings_test['res_id'].unique()



    newmatrix=np.zeros((n_users_train,n_items_train))
    # newmatrix=predictions_svd
    # for k in range(1,11):
    #     print(k)
    for i in testuserset:
        #print(i)
        for j in testrestset:
            if(j>=4416):
                pass
            else:
                # print('pop',newpopdict[str(j)])
                # print('res',predictions_svd[i][j])
                newpred = 1*pred[i][j] + 9*newpopdict[str(j)] #pred为餐馆，pop
                # print(predictions_svd[i][j])#相似度相加

                newmatrix[i][j]=newpred

    print(newmatrix.shape)


    totalrecall=0.0
    totalprecision = 0.0
    count = 0
    # topk=5
    totalf1=0.0
    for i in testuserset:
        userid=i
        user_ratings = newmatrix[i, :]
        # print(type(user_ratings))
        train_unkown_indices = np.where(train_matrix[i, :] == 0)[0]
        user_recommendations = user_ratings[train_unkown_indices]
        recommendset = set()
        for rest_id in user_recommendations.argsort()[-topk:][:: -1]:
            recommendset.add(rest_id)

        newtestset = set()
        testset = ratings_test[ratings_test['user_id'] == userid]['res_id']
        for i in testset:
            newtestset.add(i)

        # print(recommendset,newtestset)
        commonset = recommendset & newtestset
        precision = len(commonset) / topk
        recall = len(commonset) / len(newtestset)
        totalrecall += recall
        totalprecision = totalprecision + precision
        if (precision + recall == 0):
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        totalf1 = totalf1 + f1
        count = count + 1

        # print()
    print(totalprecision / count)
    print(totalrecall / count)
    # print("f1")
    print(totalf1 / count)



if __name__ == '__main__':
    for i in range(1,6):
        print(i)
        sop(i)