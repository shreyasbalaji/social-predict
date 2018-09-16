import multiprocessing as mp
import numpy as np
import vectorize as vec
import os

#generates data matrices and stores them in respective directory
def saveDataFiles(rootdir):
	dir_list = []
	for subdir, dirs, files in os.walk(rootdir):
		for d in dirs:
			dir_list.append(d)
		
	def makeMatrix(d):
		matrix = []
		for s, di, f in os.walk(rootdir + "/" + d):
			for a in f:
				if a.endswith(".json"):
					row = np.array(vec.process_json(os.path.join(rootdir, d, a)))
					matrix.append(row)
		np.save(os.path.join(rootdir, d)+'/data_matrix_'+d,np.array(matrix))


	processes = [mp.Process(target=makeMatrix, args=(dir_name,)) for dir_name in dir_list]

	for p in processes:
	    p.start()

	for p in processes:
	    p.join()

