import pickle

def read_pickle(path):
    with open(path, 'rb')as f:
        feats = pickle.load(f)
    return feats

def save_pickle(path, feats):
    with open(path, 'wb')as f:
        pickle.dump(feats, f)