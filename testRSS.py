import math




method_type = ''
def avg(dataset):
    return (sum(dataset))/(len(dataset))

def get_rss(method_type,x):
    if method_type == 'raw':
        mean = avg(data)
        rss = sum( math.pow(d-mean,2) for d in data)

    if method_type == 'split':
        s = x
        dataset1 = data[:s]
        dataset2 = data[s:]
        mean1 = avg(dataset1) if len(dataset1) > 0 else 0
        mean2 = avg(dataset2) if len(dataset2) > 0 else 0
        rss = sum( math.pow(d-mean1,2) if len(dataset1) > 0 else 0 for d in dataset1) + \
              sum( math.pow(d-mean2,2) if len(dataset2) > 0 else 0 for d in dataset2)
    return rss


if __name__ == '__main__':
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 45, 191, 234]
    print (len(data))
    for i in range(len(data)):
        rss = get_rss('split',i)
        print (rss,'', i)