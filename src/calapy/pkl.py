import pickle
import os


def save(data, directory_file):
    root = os.path.dirname(directory_file)
    if not os.path.isdir(root):
        os.makedirs(root)
    outfile = open(directory_file, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    return None


def load(directory_file):
    outfile = open(directory_file, "rb")
    data = pickle.load(outfile)
    outfile.close()
    return data

