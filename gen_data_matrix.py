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
                if a.endswith(".json") and (not a.startswith("meta")):
                    row = np.array(vec.process_json(os.path.join(rootdir, d, a)))
                    matrix.append(row)

        np.save(os.path.join(rootdir, d, 'data_matrix.npy'), np.array(matrix))

    processes = [mp.Process(target=make_matrix, args=(dir_name,)) for dir_name in dir_list]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

def save_data(subdir):
    def make_matrix(d):
        matrix = []
        count = 0
        for s, di, f in os.walk(subdir):
            for a in f:
                if a.endswith(".json") and (not a.startswith("meta")):
                    count += 1
                    row = np.array(vec.process_json(os.path.join(subdir, a)))
                    matrix.append(row)
                    if (count % 10 == 0):
                        print(count)

        np.save(os.path.join(subdir, 'data_matrix.npy'), np.array(matrix))

    make_matrix(subdir)

def load_data_matrix(subdir):
    result = np.load(os.path.join(subdir, 'data_matrix.npy'))
    if len(result.shape) == 1:
        new_result = np.empty((len(result), 394), dtype=np.float32)
        for i in range(len(result)):
            new_result[i] = result[i]
        return new_result
    return result

if __name__ == '__main__':
    # save_data_files('nlp_test_data')
    import sys
    print('Processing:', sys.argv[1])
    save_data(sys.argv[1])
