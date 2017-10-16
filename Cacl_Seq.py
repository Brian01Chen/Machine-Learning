

def sequence(c1,c2,seq,step):
    iflag = 0
    for i in range(1,step):
        if seq[i] > c1+seq[0] and iflag == 0:
            iflag = 1
        if i >1 :
            for j in range (1,i):
                for k in range (j+1,i):
                    if seq[j] - seq[k] < c2:
                        continue
                    else:
                        iflag = 2
    print (iflag)
    # iflag = 0 ,表示第一条件不满足;iflag = 2,表示第二条件不满足;iflag = 1,表示在迭代范围，条件都满足。
    if iflag in (0,2):
        return False
    else: return True

if __name__ == '__main__':
    c1,c2 = 70,10
    seq1 = [3,4,39,25,12,311,231,553,123,5112,123,45,67,78,11,234,245,12]
    seq2 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    seq3 = [3,20,25,33,41,50,59,65,72,80,88,97,106,117,17,18,19,20,21]
    step = 10
    result = sequence(c1,c2,seq3,step)
    print (result)