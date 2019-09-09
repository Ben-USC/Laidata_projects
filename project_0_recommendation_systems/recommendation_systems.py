"""
Beibei Wang
INF-553 homework 3, task 2
"""
import math
import time
from sys import argv
from pyspark.context import SparkContext
## ------ mllib is for case 1 only -----
from pyspark.mllib.recommendation import ALS, Rating#, MatrixFactorizationModel

start = time.time()

sc =  SparkContext('local[*]')
#train_file, test_file, output_file = argv[1], argv[2], argv[4]
#case_id = int(argv[3])
train_file, test_file = "./dataset/yelp_train.csv", "./dataset/yelp_val.csv"
case_id = 1
output_file = "./newest_output_task2_case"+str(case_id)+".txt"

### ----------------- preprocessing the training and testing data file --------
orig_train_rdd = sc.textFile(train_file).map(lambda l: l.split(","))
orig_test_rdd = sc.textFile(test_file).map(lambda l: l.split(","))
## remove header in both files:
header = orig_train_rdd.first()
train_rdd = orig_train_rdd.filter(lambda l: l[0] != header[0])
test_rdd = orig_test_rdd.filter(lambda l: l[0] != header[0])
## transform to rdd about unique users or businesses
user_train = train_rdd.map(lambda l: (l[0], (l[1], l[2]))).groupByKey()
busi_train = train_rdd.map(lambda l: (l[1], (l[0], l[2]))).groupByKey()
## collect user ids, user numbers, business ids, business numbers
## already checked, no duplicates for each user, or each business
user_ids = user_train.keys().collect()
user_num = len(user_ids)
busi_ids = busi_train.keys().collect()
busi_num = len(busi_ids)
## create dictionaries, map id strings to integers:
def create_dic(lst):
    dic = {}
    for i in range(len(lst)):
        dic[lst[i]] = i
    return dic
## create (business_id: index) and (user_id: index)dictionary
## note that in the yelp_val.csv, sef of users is the same as in yelp_train.csv
## in the yelp_val.csv, sef of businesses is not the same as in yelp_train.csv
busi_dict = create_dic(busi_ids)
user_dict = create_dic(user_ids)
## transfer rdd into ( (user_id, {busi_id:rating, ...}), ...)
user_dict_train = user_train.map(lambda l: (user_dict[l[0]], {busi_dict[t[0]]:float(t[1]) for t in l[1]}))
user_dict_train = user_dict_train.collect()
## transfer rdd into ( (busi_id, {user_id:rating, ...}), ...)
busi_dict_train = busi_train.map(lambda l: (busi_dict[l[0]], {user_dict[t[0]]:float(t[1]) for t in l[1]}))
busi_dict_train = busi_dict_train.collect()

## create *_tot list[(total_rating, num_of_ratings), ... ]:
def find_total(dic, num):
    tot = [0.0] * num
    for u in dic:
        lst = u[1].values()
        tot[u[0]] = (sum(lst), len(lst))
    return tot
user_tot = find_total(user_dict_train, user_num)
busi_tot = find_total(busi_dict_train, busi_num)
## create *_ave list[ ave1, ave2, ... ]:
def find_ave(dic, num):
    ave = [0.0] * num
    for u in dic:
        lst = u[1].values()
        ave[u[0]] = sum(lst)/len(lst)
    return ave
user_ave = find_ave(user_dict_train, user_num)
busi_ave = find_ave(busi_dict_train, busi_num)

### ----------------- case 2: User-based CF -------------
class User_Based_CF(object):
    
    ## compute pearson correlation given user u and v
    ## u and v has the format (user_id, {busi_id:rating, ...})
    def W(self, u, v):
        if u == v:
            return 1.0
        inter = u[1].keys() & v[1].keys()
        if len(inter) == 0:
            return 1.0
        l1 = [ u[1][i] for i in inter ]
        ave1 = sum(l1)/len(l1)
        l2 = [ v[1][i] for i in inter ]
        ave2 = sum(l2)/len(l2)
        l1 = [ r-ave1 for r in l1 ]
        l2 = [ r-ave2 for r in l2 ]
        numerator = sum([i*j for i, j in zip(l1, l2)])
        if numerator == 0:
            return 1.0
        domi1 = math.sqrt(sum([r**2 for r in l1]))
        domi2 = math.sqrt(sum([r**2 for r in l2]))
        return numerator/(domi1*domi2)

    ## define function to calculate r_ub given u, b
    ## u, b are integers here
    def find_rub(self, u, b):
        if u not in busi_dict_train[b][1]:
            numerator = user_tot[u][0]
            dominator = user_tot[u][1]
        else:        
            numerator = user_tot[u][0] - busi_dict_train[b][1][u]
            dominator = user_tot[u][1] - 1
        if dominator == 0:
            return 0.0
        else:
            return numerator / dominator

    ## define function to predict rating given user_id and busi_id
    def pred(self, pair):
        ## new user case:
        if pair[0][0] not in user_dict:
            return pair[0], 3.5
        a = user_dict[pair[0][0]]
        ## new business case:
        if pair[0][1] not in busi_dict:
            return pair[0], user_ave[a]
        ## otherwise:
        b = busi_dict[pair[0][1]]
        ## calculate P_ub (rating of user u on business b):
        ra = self.find_rub(a, b)
        u_lst = busi_dict_train[b][1].keys()
        r_lst = busi_dict_train[b][1].values()
        ru_lst = [self.find_rub(u_id, b) for u_id in u_lst]
        r_ru = [r - ru for r, ru in zip(r_lst, ru_lst)]
        w_lst = [self.W(user_dict_train[a], user_dict_train[u_id]) for u_id in u_lst]
        dominator = sum([abs(w) for w in w_lst])
        if dominator == 0:
            return pair[0], ra
        else:
            rw = [i*j for i, j in zip(r_ru, w_lst)]
            return pair[0], ra + sum(rw) / dominator

### ----------------- case 3: Item-based CF -------------
class Item_Based_CF(object):
    
    ## compute pearson correlation given item bi, bj
    ## bi and bj has the format (busi_id, {user_id:rating, ...})
    def W(self, bi, bj):
        inter = bi[1].keys() & bj[1].keys()
        if len(inter) == 0:
            return 0.5
        
        ri = [ bi[1][i] for i in inter ]
        ri_ave = busi_ave[bi[0]]
        ri_ = [ r - ri_ave for r in ri ]
        
        rj = [ bj[1][j] for j in inter ]
        rj_ave = busi_ave[bj[0]]
        rj_ = [ r - rj_ave for r in rj ]
        
        numerator = sum([ r1*r2 for r1, r2 in zip(ri_, rj_) ])
        if numerator == 0.0:
            return 0.5
        domi1 = math.sqrt(sum([ r**2 for r in ri_ ]))
        domi2 = math.sqrt(sum([ r**2 for r in rj_ ]))
        return 0.5#numerator/(domi1*domi2)

    ## predict using W and ratings
    ## pair has the format: ((user_id, busi_id), rating)
    def pred(self, pair, Neigh_num):
        ## new user (not seen in the training set)
        if pair[0][0] not in user_dict:
            if pair[0][1] not in busi_dict:
                return pair[0], 3.5
            else:
                return pair[0], busi_ave[busi_dict[pair[0][1]]]
        ## old user:
        u = user_dict[pair[0][0]]
        ## new business (not seen in the training set):
        if pair[0][1] not in busi_dict:
            return pair[0], user_ave[u] # + 0.5 # 3.5
        ## old user and old business:
        b = busi_dict[pair[0][1]]
        
        ## list of items rated by u:
        b_lst = user_dict_train[u][1].keys()
        ## list of ratings by u:
        r_lst = user_dict_train[u][1].values()
        ## list of w between b and every item in b_lst
        w_lst = [ self.W(busi_dict_train[b], busi_dict_train[b_id]) for b_id in b_lst ]
        ## make a (w, r) tuple, to select top N neighbors
        w_r_tuple = [ (w, r) for w, r in zip(w_lst, r_lst) ]
        w_r_tuple = sorted(w_r_tuple, key=lambda x:x[0], reverse=True)[0: Neigh_num]
        numerator = sum([ r*w for w, r in w_r_tuple ])
        
        if numerator == 0:
            return pair[0], user_ave[u]
        else:
            dominator = sum([ abs(w) for w,r in w_r_tuple ])#*fac
            rating = numerator/dominator #+ pair[1]*0.02
            #rating = rating if rating <= 5.0 else 5.0
            rating = busi_ave[b]
            return pair[0], rating#round(rating,3)

## 1. evaluate the model on training set:
#train_trans_rdd = train_rdd.map(lambda l: ((l[0], l[1]), float(l[2])))
#train_pred_rdd = train_trans_rdd.map(pred)
#train_true_pred = train_trans_rdd.join(train_pred_rdd)
#train_RMSE = train_true_pred.map(lambda l: (l[1][0] - l[1][1])**2).sum()
#train_RMSE = math.sqrt(train_RMSE/train_rdd.count())
#print(f"RMSE on training set = {train_RMSE}")     

if case_id != 1:
    test_trans_rdd = test_rdd.map(lambda l: ((l[0], l[1]), float(l[2])))
    if case_id == 2:
        ## user-based CF
        test_pred_rdd = test_trans_rdd.map(lambda pair: User_Based_CF().pred(pair))
    elif case_id == 3:
        ## item-based CF
        Neigh_num = 11 ## number of neighbors
        test_pred_rdd = test_trans_rdd.map(lambda pair: Item_Based_CF().pred(pair, Neigh_num))
    test_true_pred = test_trans_rdd.join(test_pred_rdd).collect()

else:
    ### ----------------- case 1: SparkML ----------
    ## format the data to be user(int), business(int) and ratings
    user_ids_test = test_rdd.map(lambda l: (l[0], l[1])).groupByKey().keys().collect()
    busi_ids_test = test_rdd.map(lambda l: (l[1], l[0])).groupByKey().keys().collect()
    ## create busi_dict_all to include all businesses
    busi_ids_all = busi_ids[:]
    for bid in busi_ids_test:
        if bid not in busi_dict:
            busi_ids_all.append(bid)
    busi_dict_all = create_dic(busi_ids_all)
    ## create user_dict_all to include all users
    user_ids_all = user_ids[:]
    for uid in user_ids_test:
        if uid not in user_dict:
            user_ids_all.append(uid)
    user_dict_all = create_dic(user_ids_all)
    
    ## extract new businesss and new users:
    new_pair = []
    test_lst = test_rdd.collect()
    for pair in test_lst:
        if pair[0] not in user_dict or pair[1] not in busi_dict:
            new_pair.append(((user_dict_all[pair[0]], busi_dict_all[pair[1]]), (float(pair[2]), 3.7)))
    
    ## transform the train and test data
    train_data = train_rdd.map(lambda l: Rating(user_dict_all[l[0]], busi_dict_all[l[1]], float(l[2])))
    test_data = test_rdd.map(lambda l: Rating(user_dict_all[l[0]], busi_dict_all[l[1]], float(l[2])))

    # Build the recommendation model using ALS
    #rank, iterations = 2, 5
    
    ranks = [2,4,6,8,10]
    iterations = [5, 10]
    
    for rank in ranks:
        for itr in iterations:
            model = ALS.train(train_data, rank, itr, seed=16807)
            # Evaluate the model
            test_data_for_pred = test_data.map(lambda l: (l[0], l[1]))
            #test_pred_data = test_data_eval.map(lambda pair: model.predict(pair[0], pair[1]))#.map(lambda l: ((l[0], l[1]), l[2]))
            test_pred_data = model.predictAll(test_data_for_pred).map(lambda l: ((l[0], l[1]), l[2] if l[2] <5.0 else 5.0))
            test_true_pred = test_data.map(lambda l: ((l[0], l[1]), l[2])).join(test_pred_data).collect()
            for pair in new_pair:
                test_true_pred.append(pair)
            ## transform from (num, num) to (string, string)
            test_true_pred = [((user_ids_all[t[0][0]], busi_ids_all[t[0][1]]), (t[1][0], t[1][1])) for t in test_true_pred]

            ## compute RMSE on validation(test) dataset: 
            test_RMSE = sum([ (pair[1][0]-pair[1][1])**2 for pair in test_true_pred ])
            test_RMSE = math.sqrt(test_RMSE/len(test_true_pred))
            print(f"RMSE on test set for case {case_id}, rank {rank}, iteration {itr} is: {test_RMSE}")

## ----------------------- write out to file: ---------------
f = open(output_file, "w")
f.write("user_id, business_id, prediction\n")
## test_true_pred has the format: [((u_id, busi_id), (true, pred)), ...]
for pair in test_true_pred:
    f.write(f"{pair[0][0]},{pair[0][1]},{pair[1][1]}\n")
f.close()

end = time.time()
print(f"Duration for case {case_id}: {int(end-start)}") 
