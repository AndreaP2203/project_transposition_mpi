In order to execute the program mpi_matrix_transposition.c, inside the cluster login node, a folder named project_mpi within the file mpi_matrix_transposition.c and the file project_pbs.pbs. Then, inside the folder send the execution command qsub project_pbs.pbs .

The .pbs file does on his own the execution of the program for each number of processes and N, the results are stored automatically in files named results_N_np.txt .

For example to see the project_transposition.c results, executed with a size matrix N=1024 and 16 np, the name file will be results_1024_16.txt .
