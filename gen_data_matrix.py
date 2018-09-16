import multiprocessing as mp
import numpy as np
import vectorize as vec
import os

# generates data matrices and stores them in respective directory
def save_data_files(rootdir):
    dir_list = []
    for subdir, dirs, files in os.walk(rootdir):
        for d in dirs:
            dir_list.append(d)
            
    def make_matrix(d):
        matrix = []
        for s, di, f in os.walk(rootdir + "/" + d):
            for a in f:
                if a.endswith(".json"):
                    row = np.array(vec.process_json(os.path.join(rootdir, d, a)))
                    matrix.append(row)

        np.save(os.path.join(rootdir, d, 'data_matrix.npy'), np.array(matrix))

    processes = [mp.Process(target=make_matrix, args=(dir_name,)) for dir_name in dir_list]

    for p in processes:
        p.start()

    for p in processes:
        p.join()


def load_data_matrix(subdir):
    return np.load(os.path.join(subdir, 'data_matrix.npy'))
