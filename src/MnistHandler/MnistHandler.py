import cPickle, gzip

# Load the dataset
def readMNISTData(path):
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set,valid_set,test_set
