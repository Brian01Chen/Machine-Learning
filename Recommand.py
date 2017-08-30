import codecs
from math import sqrt
users = {
"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0,"Norah Jones": 4.5, "Phoenix": 5.0,"Slightly Stoopid": 1.5,"The Strokes": 2.5, "Vampire Weekend": 2.0},
"Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5,"Deadmau5": 4.0, "Phoenix": 2.0,"Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
"Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0,"Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5,"Slightly Stoopid": 1.0},
"Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0,"Deadmau5": 4.5, "Phoenix": 3.0,"Slightly Stoopid": 4.5, "The Strokes": 4.0,"Vampire Weekend": 2.0},
"Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0,"Norah Jones": 4.0, "The Strokes": 4.0,"Vampire Weekend": 1.0},
"Jordyn": {"Broken Bells": 4.5, "Deadmau5": 4.0,"Norah Jones": 5.0, "Phoenix": 5.0,"Slightly Stoopid": 4.5, "The Strokes": 4.0,"Vampire Weekend": 4.0},
"Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0,"Norah Jones": 3.0, "Phoenix": 5.0,"Slightly Stoopid": 4.0, "The Strokes": 5.0},
"Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0,"Phoenix": 4.0, "Slightly Stoopid": 2.5,"The Strokes": 3.0}
}


#基于物品的协同过滤：修正余弦计算物品之间的相似度
def calcGoodsSimilar(brand1,brand2):
    ratings_average = {}.fromkeys(users.keys())
    for ur in ratings_average:
        ratings_average[ur] = round(sum(users[ur].values())/len(users[ur]),2)
    print (ratings_average)

    diff,sum_b_1,sum_b_2 = 0,0,0
    for (user,ratings) in users.items():
        u_avg = ratings_average[user]
        if brand1 in ratings and brand2 in ratings:
            diff += (ratings[brand1]-u_avg)*(ratings[brand2]-u_avg)
            sum_b_1 += pow((ratings[brand1]-u_avg),2)
            sum_b_2 += pow((ratings[brand2]-u_avg),2)
    dis_1_2 = diff/(sqrt(sum_b_1)*sqrt(sum_b_2))
    print (dis_1_2)
    
calcGoodsSimilar('Blues Traveler','Norah Jones')


# 闵科夫斯基距离，欧几里得距离，余弦相似度，皮尔逊相关系数，K临近算法

def userRatings(userid):
    ratings = users[userid]
    ratings = [item for item in ratings.items()]
    ratings = sorted(ratings,key=lambda art : art[1],reverse = True)
    print (ratings)
    return ratings
    
#userRatings('Angelica')

def MinKfskDis(rating1,rating2,p):
    distance = 0
    for k in rating1:
        if k in rating2:             
            distance += pow((abs(rating1[k]-rating2[k])),p)
    minksfskdistance = pow(distance,1/p)
    print (minksfskdistance)
    return minksfskdistance

#MinKfskDis(users['Angelica'],users['Bill'],1)

def CosDis(rating1,rating2):
    distance = 0
    for k in rating1:
        if k in rating2:
            distance += abs(rating1[k]*rating2[k])
    var1 =  pow(sum(map(lambda x:pow(x,2), [v for v in rating1.values()])),1/2)
    var2 =  pow(sum(map(lambda x:pow(x,2), [v for v in rating2.values()])),1/2)
    cosdis = round(distance/(var1*var2),2)
    return cosdis

#CosDis(users['Angelica'],users['Bill']) 

def pearson(rating1,rating2):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x, 2)
            sum_y2 += pow(y, 2)
    if n == 0:
        return 0
    # 计算分母
    denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n)
                   * sqrt(sum_y2 - pow(sum_y, 2) / n))
    if denominator == 0:
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) / n) / denominator

def KNNeighbor(username,k):
    '''获取邻近用户'''
    piot_rating = users[username]
    weight_rating_list = []
    for user in users:
        if user != username:
            weight_rating_list.append((user,pearson(piot_rating,users[user])))
    pearson_rating_list = sorted(weight_rating_list,key=lambda wrat:wrat[1],reverse=True)
    KNNeighbors = pearson_rating_list[:k]
    #print (KNNeighbors)
    sum_pearson = 0
    for k in KNNeighbors:
        sum_pearson += k[1]
    #取权重
    weight_rating_list = [(knn[0],round(knn[1]/sum_pearson,2)) for knn in KNNeighbors]
    #print (weight_rating_list)
    return weight_rating_list
    
#KNNeighbor('Veronica',3)

def recommender(userid,distype):
    rating = users[userid]
    maxdis = 0
    for user in users:
        if user != userid:
            if CosDis(rating,users[user]) > maxdis:
                maxdis = CosDis(rating,users[user])
                simiuser = user
    #获取最近的user
    bestuser = simiuser
    recommender = [(k,users[bestuser][k]) for k in users[bestuser] if k not in rating]
    print ('Best User: %s\nBand Recommender: %s' %(bestuser,recommender))
    return bestuser,recommender

#recommender('Veronica',1)

