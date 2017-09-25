import math


data = [1,2,3,4,5,6,7,8,9,10,31,32,33,45,191,234]

method_type = ''
def avg(*dataset):
    return (sum(dataset))/(len(dataset))

def get_rss(method_type):
    if method_type == 'raw':
        mean = avg(data)
        rss = sum( math.pow(d-mean,2) for d in data)
        return rss


if __name__ == '__main__':
    rss = get_rss(raw)
    print (rss)