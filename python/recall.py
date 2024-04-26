import math
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import os
import csv
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp
from scipy.sparse.linalg import svds

parentdic = os.path.dirname(os.getcwd())


#首先通过ratings_100划分测试集和训练集合
def traintest():

    data = pd.read_csv("ratings_100.csv",
                       names=['user_id', 'res_id', 'rating'])

    userset = set()
    for i in range(len(data['user_id'])):
        userset.add(data["user_id"][i])  # 把用户Id提取出来，得到用户id 的set

    traindata = pd.DataFrame(columns=['user_id', 'res_id', 'rating'])
    testdata = pd.DataFrame(columns=['user_id', 'res_id', 'rating'])

    for i in userset:
        newdata = data[data['user_id'] == i]
        newdata = shuffle(newdata) #打乱数据
        m = len(newdata)
        traindatalen = round(m * 0.8)
        x_train = newdata[0:traindatalen]
        x_test = newdata[traindatalen:m]

        traindata = pd.concat([traindata, x_train])
        testdata = pd.concat([testdata, x_test])

    traindata.to_csv("train100.csv", index=0, header=None)
    testdata.to_csv("test100.csv", index=0, header=None)

# traintest()

def dataanalysis():
    rs_cols = ['user_id', 'res_id', 'rating']  #train和test里面包含的列
    ratings_train = pd.read_csv(parentdic+'/data/train100.csv', names=rs_cols,index_col=False)
    ratings_test = pd.read_csv(parentdic + '/data/test100.csv', names=rs_cols)

    n_users= len(ratings_train['user_id'].unique())  #统计得到用户数目3291个
    n_items= len(ratings_train['res_id'].unique()) #统计得到餐厅数目4088个

    #重新编号，使得train里面的userid从0开始，res_id也从0开始，从而可以更好的构建矩阵

    user2iddict=dict()
    usercount=0
    rest2iddict=dict()
    restcount=0
    with open(parentdic + "/data/train100.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            userid = data[0]
            res_id=data[1]
            rating=data[2]
            if(user2iddict.__contains__(userid)):
                pass
            else:
                user2iddict.__setitem__(userid,usercount)
                usercount=usercount+1
            if(rest2iddict.__contains__(res_id)):
                pass
            else:
                rest2iddict.__setitem__(res_id,restcount)
                restcount=restcount+1

    with open(parentdic + "/data/test100.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            userid = data[0]
            res_id=data[1]
            rating=data[2]
            if(user2iddict.__contains__(userid)):
                pass
            else:
                user2iddict.__setitem__(userid,usercount)
                usercount=usercount+1
            if(rest2iddict.__contains__(res_id)):
                pass
            else:
                rest2iddict.__setitem__(res_id,restcount)
                restcount=restcount+1

    with open(parentdic + "/data/train100.csv", 'r', encoding='unicode_escape') as f,open(parentdic + "/data/newtrain100.csv", 'w', encoding='utf-8',
                      newline='') as out:
        writer = csv.writer(out)
        reader = csv.reader(f)
        for data in reader:
            userid = data[0]
            res_id = data[1]
            rating = data[2]
            writer.writerow([user2iddict[userid],rest2iddict[res_id],rating])

    with open(parentdic + "/data/test100.csv", 'r', encoding='unicode_escape') as f,open(parentdic + "/data/newtest100.csv", 'w', encoding='utf-8',
                      newline='') as out:
        writer = csv.writer(out)
        reader = csv.reader(f)
        for data in reader:
            userid = data[0]
            res_id = data[1]
            rating = data[2]
            writer.writerow([user2iddict[userid],rest2iddict[res_id],rating])

    print(usercount)
    print(restcount)

#dataanalysis()


def counter():
    rs_cols = ['user_id', 'res_id', 'rating']  # train和test里面包含的列
    ratings_test = pd.read_csv('data/newtest100.csv', names=rs_cols, index_col=False, header=None)
    testuserset=ratings_test['user_id']

    dicts = {}  # 存放统计结果
    for li in testuserset:
        if li in testuserset:
            if dicts.get(li):  # 再次统计
                dicts.update({li: dicts.get(li) + 1})
            else:
                dicts.update({li: 1})
    return dicts



def ucf(topk,n_similar):

    rs_cols = ['user_id', 'res_id', 'rating']  # train和test里面包含的列
    ratings_train = pd.read_csv('data/newtrain100.csv', names=rs_cols, index_col=False, header=None)
    ratings_test = pd.read_csv('data/newtest100.csv', names=rs_cols, index_col=False, header=None)

    n_users_train = len(ratings_train['user_id'].unique())  # 统计得到用户数目3291个
    n_items_train= len(ratings_train['res_id'].unique())

    n_users_test = len(ratings_test['user_id'].unique())  # 统计得到用户数目3291个
    n_items_test = len(ratings_test['res_id'].unique())

    # 统计得到用户数目3291个
    # 统计得到餐厅数目4088个 实际test里面还有一些train里面从未出现过的res_id
                                                    #一共4417个

    train_matrix = np.zeros((n_users_train, n_items_train))
    for line in ratings_train.itertuples():
        train_matrix[line[1], line[2]] = line[3]

    user_similarity = pairwise_distances(train_matrix, metric='cosine')

    # n_similar = 1
    similar_n = user_similarity.argsort()[:, -n_similar:][:, ::-1] #找出各个user前30个相似用户
    #print(similar_n.shape)
    pred = np.zeros((n_users_train, n_items_train))

    for i, users in enumerate(similar_n):
        similar_users_indexes = users
        similarity_n = user_similarity[i, similar_users_indexes] #相似度
        matrix_n = train_matrix[similar_users_indexes, :]
        rated_items = similarity_n[:, np.newaxis].T.dot(
            matrix_n - matrix_n.mean(axis=1)[:, np.newaxis]) / similarity_n.sum()
        pred[i, :] = rated_items

    testuserset=ratings_test['user_id'].unique()

    totalprecision=0.0
    count=0
    totalF1=0
    dicts=counter()
    totalrecall=0.0
    for i in testuserset:
        userid=i #是测试集合中的用户id
        #print(userid)
        user_ratings = pred[userid, :]
        train_unkown_indices = np.where(train_matrix[userid, :] == 0)[0]
        user_recommendations = user_ratings[train_unkown_indices]
        recommendset=set()
        for res_id in user_recommendations.argsort()[-topk:][:: -1]:
            recommendset.add(res_id)

        newtestset=set()
        testset=ratings_test[ratings_test['user_id']==userid]['res_id']
        for i in testset:
            newtestset.add(i)

        #print(recommendset,newtestset)
        commonset = recommendset & newtestset
        precision = len(commonset) / topk
        recall = len(commonset) / dicts[userid]
        totalrecall +=recall
        totalprecision = totalprecision + precision
        if (precision+recall==0):
            f1=0
        else:
            f1=2*precision*recall/(precision+recall)
        totalF1=f1+totalF1
        count=count+1

    # print()
    # print(totalprecision/count)
    # print("召回率")
    # print(totalrecall/count)
    print(totalF1/count)

# ucf(3,4)

def rcf(topk,n_similar):

    rs_cols = ['user_id', 'res_id', 'rating']  # train和test里面包含的列
    ratings_train = pd.read_csv('data/newtrain100.csv', names=rs_cols, index_col=False, header=None)
    ratings_test = pd.read_csv('data/newtest100.csv', names=rs_cols, index_col=False, header=None)

    n_users_train = len(ratings_train['user_id'].unique())  # 统计得到用户数目3291个
    n_items_train = len(ratings_train['res_id'].unique())

    n_users_test = len(ratings_test['user_id'].unique())  # 统计得到用户数目3291个
    n_items_test = len(ratings_test['res_id'].unique())

    # 统计得到用户数目3291个
    # 统计得到餐厅数目4088个 实际test里面还有一些train里面从未出现过的res_id
    # 一共4417个

    train_matrix = np.zeros((n_users_train, n_items_train))
    for line in ratings_train.itertuples():
        train_matrix[line[1], line[2]] = line[3]

    # n_similar=1
    item_similarity = pairwise_distances(train_matrix.T, metric='cosine')
    similar_n = item_similarity.argsort()[:, -n_similar:][:, ::-1]
    #print('similar_n shape: ', similar_n.shape)
    pred = np.zeros((n_users_train, n_items_train))

    for i, items in enumerate(similar_n):
        similar_items_indexes = items
        similarity_n = item_similarity[i, similar_items_indexes]
        matrix_n = train_matrix[:, similar_items_indexes]
        rated_items = matrix_n.dot(similarity_n) / similarity_n.sum()
        pred[:, i] = rated_items

    testuserset = ratings_test['user_id'].unique()

    totalprecision = 0.0
    count = 0
    totalrecall=0.0
    totalf1=0.0
    for i in testuserset:
        userid = i  # 是测试集合中的用户id
        # print(userid)
        user_ratings = pred[userid, :]
        train_unkown_indices = np.where(train_matrix[userid, :] == 0)[0]
        user_recommendations = user_ratings[train_unkown_indices]
        recommendset = set()
        for res_id in user_recommendations.argsort()[-topk:][:: -1]:
            recommendset.add(res_id)

        newtestset = set()
        testset = ratings_test[ratings_test['user_id'] == userid]['res_id']
        for i in testset:
            newtestset.add(i)

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
    # print(totalrecall/count)
    # print("f1")
    # print(totalf1/count)


def svd(topk,k):
    rs_cols = ['user_id', 'res_id', 'rating']  # train和test里面包含的列
    ratings_train = pd.read_csv('data/newtrain100.csv', names=rs_cols, index_col=False, header=None)
    ratings_test = pd.read_csv('data/newtest100.csv', names=rs_cols, index_col=False, header=None)

    n_users_train = len(ratings_train['user_id'].unique())  # 统计得到用户数目3291个
    n_items_train = len(ratings_train['res_id'].unique())

    n_users_test = len(ratings_test['user_id'].unique())  # 统计得到用户数目3291个
    n_items_test = len(ratings_test['res_id'].unique())

    # 统计得到用户数目3291个
    # 统计得到餐厅数目4088个 实际test里面还有一些train里面从未出现过的res_id
    # 一共4417个

    train_matrix = np.zeros((n_users_train, n_items_train))
    for line in ratings_train.itertuples():
        train_matrix[line[1], line[2]] = line[3]

    u, s, vt = svds(train_matrix, k) #k是分解的纬度
    s_diag_matrix = np.diag(s)
    predictions_svd = np.dot(np.dot(u, s_diag_matrix), vt)

    testuserset = ratings_test['user_id'].unique()

    totalprecision = 0.0
    count = 0
    dicts = counter()
    totalrecall = 0.0
    totalF1=0
    for i in testuserset:

        userid = i  # 是测试集合中的用户id

        user_ratings = predictions_svd[i, :]
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
        totalprecision = totalprecision + precision
        recall = len(commonset) /len(newtestset)
        totalrecall += recall
        # print("useid",userid)
        # print("....")
        # print(len(commonset))
        # print(dicts[userid])
        # print("....")
        if (precision + recall!=0):
            F1 = 2 * (precision * recall) / (precision + recall)
        else:
            F1=0
        totalF1 += F1

        count = count + 1

    # print()
    # print(totalprecision / count)
    # print("召回率")
    # print(totalrecall/count)
    print(totalF1/count)

# svd(topk,k)
#
def pop(topk):

    rs_cols = ['user_id', 'res_id', 'rating']  # train和test里面包含的列
    ratings_train = pd.read_csv('data/newtrain100.csv', names=rs_cols, index_col=False, header=None)
    ratings_test = pd.read_csv('data/newtest100.csv', names=rs_cols, index_col=False, header=None)

    n_users_train = len(ratings_train['user_id'].unique())  # 统计得到用户数目3291个
    n_items_train = len(ratings_train['res_id'].unique())

    popdict=dict()
    with open('data/newtrain100.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            userid = data[0]
            res_id=data[1]
            rating=data[2]
            if(popdict.__contains__(res_id)):
                popdict[res_id]=popdict[res_id]+1
            else:
                popdict.__setitem__(res_id,1)

    #作归一化，使得值在0-1之间

    maxvalue = max(popdict.values())
    newpopdict=dict()
    for k,v in popdict.items():
        newpopdict.__setitem__(k,v/maxvalue)

    newpopdict = sorted(newpopdict.items(), key=lambda x: x[1], reverse=True)

    testuserset = ratings_test['user_id'].unique()

    totalprecision = 0.0
    count = 0
    totalF1=0.0
    dicts = counter()
    totalrecall = 0.0
    for i in testuserset:
        userid = i  # 是测试集合中的用户id

        newpopdictsimi = newpopdict[:topk]
        recommendset = set()
        for i in newpopdictsimi:
            recommendset.add(int(i[0]))

        newtestset = set()
        testset = ratings_test[ratings_test['user_id'] == userid]['res_id']
        for i in testset:
            newtestset.add(i)

        # print(recommendset,newtestset)

        # print(recommendset,newtestset)
        commonset = recommendset & newtestset
        precision = len(commonset) / topk
        totalprecision = totalprecision + precision
        #testsetlen=int(dicts[userid])
        recall=len(recommendset)/len(newtestset)
        #print('userid',userid,dicts[userid])
        totalrecall +=recall

        F1 = 2 * (precision * recall) / (precision + recall)
        totalF1+=F1
        count = count + 1

    # print()
    # print(totalprecision / count)
    # print("召回率")
    # #print(totalrecall)
    # print(totalrecall/count)
    # print("F1")
    print(totalF1/count)

# pop(topk)



if __name__ == '__main__':
    # topk = 3
    # n_similar =70
    # k=80
    #
    # for i in range(1,6):
    #     topk=i
    #     # print(i)
    #     # print("ucf")
    #     ucf(topk,n_similar)
    #     print("rcf")
    #     rcf(topk, n_similar)
    #     print()
    print()
    print("ref")
    # for i in range(1, 6):
    #     topk = i
    topk=5
    n_similar=4
    rcf(topk, n_similar)

    # for i in range(1, 6):
        # topk=i
        # k=100
        # svd(topk, k)
        # print("pop")
        # pop(topk)
    # print("topk:" + str(topk) + "    邻居个数n_similar:" + str(n_similar) + "  分解维度k:" + str(k))









