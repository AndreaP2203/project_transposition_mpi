#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

float random_float() {
    return (float) (rand() % 10);
}

// SEQUENTIAL TRANSPOSITION

void matrix_init(float **matrix, int n) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            matrix[i][j] = random_float();
        }
    }
}

int checkSym(float **matrix, int n) {
    int x=0; 
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            if(matrix[i][j] == matrix[j][i]) {
                x++;
            }
        }
    }
    return x;
}

void matTranspose(float **matrix, float** matrix_transposed, int n) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            matrix_transposed[j][i] = matrix[i][j];
        }
    }
}

// MPI PARALLELIZATION

int checkSymMPI(float **matrix, int n, int rank, int size) {
    int rows_per_process = n/size;
    int local_count=0, global_count=0;

    float *matrix_flat = NULL;

    if(rank==0) {
        matrix_flat = (float*) malloc (n*n*sizeof(float));
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                matrix_flat[i*n+j] = matrix[i][j];
            }
        }
    } 

    float *local_matrix = (float*)malloc(rows_per_process*n*sizeof(float));

    MPI_Scatter(matrix_flat, rows_per_process*n, MPI_FLOAT, local_matrix,
                rows_per_process*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(rank != 0) {
        matrix_flat = (float*) malloc (n*n*sizeof(float));
    }

    MPI_Bcast(matrix_flat, n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for(int i=0; i<rows_per_process; i++) {
        int global_row = rank * rows_per_process + i;
        for(int j=0; j<n; j++) {
            if(local_matrix[i*n+j] == matrix_flat[i*n+global_row]) {
                local_count++;
            }
        }
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local_matrix);

    if(rank == 0) {
        free(matrix_flat);
    }

    if(rank==0) {
        return global_count;
    } else {
        return -1;
    }

}

void matTransposeMPI(float **matrix, float **matrix_transposed, int n, int rank, int size) {
    int local_rows = n/size;

    float *matrix_transposed_flat = NULL;
    float *matrix_flat = NULL;

    if(rank == 0) {
        matrix_flat = (float*) malloc (n*n*sizeof(float));
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                matrix_flat[i*n+j] = matrix[i][j];
            }
        }
        matrix_transposed_flat = (float*) malloc (n*n*sizeof(float));
    }

    float *local_matrix = (float*) malloc (local_rows*n*sizeof(float));
    float* local_transposed = (float*) malloc (local_rows*n*sizeof(float));

    MPI_Scatter(matrix_flat, local_rows*n, MPI_FLOAT, local_matrix,
                local_rows*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for(int i=0; i<local_rows; i++) {
        for(int j=0; j<n; j++) {
            local_transposed[j*local_rows+i] = local_matrix[i*n+j];
        }
    }

    MPI_Gather(local_transposed, local_rows*n, MPI_FLOAT, matrix_transposed_flat,
                local_rows*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        int flat_index=0;

        for(int k=0; k<size; k++) {
            for(int i=0; i<n; i++) {
                for(int j=local_rows*k; j<local_rows*(k+1); j++) {
                    matrix_transposed[i][j] =
                    matrix_transposed_flat[flat_index];
                    flat_index++;
                }
            }
        }

        free(matrix_flat);
        free(matrix_transposed_flat);
    }

    free(local_matrix);
    free(local_transposed);

}

// DEALLOCATION

void free_matrix(float **matrix, int n) {
    for(int i=0; i<n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

// INPUT CHECK

    int N;

    if(rank == 0) {
        N = atoi(argv[1]);
        int temp = N;

        while(temp != 1) {
            if(temp%2 != 0) {
                printf("N non potenza di due !!");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            temp = temp/2;
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(N%size != 0) {
        if(rank == 0) {
            printf("La dimensione N deve essere divisibile per il numero dei processi");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

// MATRIX INIZIALIZATION

    if(rank == 0) {
        printf("Numero righe N: %d \n\n", N);
    }

    float **matrix = NULL;
    float **matrix_transposed = NULL;

    if(rank == 0) {
        matrix = (float **) malloc (N*sizeof(float*));
        for(int i=0; i<N; i++) {
            matrix[i] = (float *) malloc (N*sizeof(float));
        }

        matrix_transposed = (float **) malloc (N*sizeof(float*));
        for(int i=0; i<N; i++) {
            matrix_transposed[i] = (float *) malloc (N*sizeof(float));
        }

        matrix_init(matrix, N);
    }

// SYMMETRY

    double sum_sym_seq = 0.0, sum_sym_par = 0.0;
    double avg_sym_seq, avg_sym_par;

for(int i=0; i<50; i++) {

    if(rank == 0) {
        double seq_sym_time1 = omp_get_wtime();
        if(checkSym(matrix, N) == (N*N)) {
            printf("Matrix symmetric !! \n");
            MPI_Finalize();
            return 0;
        }
        double seq_sym_time2 = omp_get_wtime();
        sum_sym_seq += (seq_sym_time2 - seq_sym_time1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double par_sym_time1 = MPI_Wtime();
    if(checkSymMPI(matrix, N, rank, size) == (N*N)) {
        printf("Matrix symmetric !! \n");
        MPI_Finalize();
        return 0;
    }
    double par_sym_time2 = MPI_Wtime();
    double sum_sym_par_locale = (par_sym_time2 - par_sym_time1);
    MPI_Reduce(&sum_sym_par_locale, &sum_sym_par, 1, MPI_DOUBLE, MPI_SUM,
                0, MPI_COMM_WORLD);

}

    if(rank == 0) {
        avg_sym_seq = sum_sym_seq/50;
        avg_sym_par = sum_sym_par/50;

        printf("AVG WTime SYM SEQ: %f seconds\n", avg_sym_seq);
        printf("AVG WTime SYM PAR: %f seconds\n", avg_sym_par);
    }

    double sum_trans_seq=0, sum_trans_par=0;
    double avg_trans_seq, avg_trans_par;

for(int i=0; i<50; i++) {

// SEQUENTIAL TRANSPOSITION

    if(rank == 0) {
        double seq_trans_time1 = omp_get_wtime();
            matTranspose(matrix, matrix_transposed, N);
        double seq_trans_time2 = omp_get_wtime();
        printf("Attempt number SEQUENTIAL: %d --------\n", i+1);
        printf("WTime in Sequential: %f seconds\n\n", seq_trans_time2-seq_trans_time1);
        sum_trans_seq += (seq_trans_time2 - seq_trans_time1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

// PARALLEL TRANSPOSITION

    double par_trans_time1 = MPI_Wtime();
        matTransposeMPI(matrix, matrix_transposed, N, rank, size);
    double par_trans_time2 = MPI_Wtime();

    double partial_sum_trans_par = (par_trans_time2 - par_trans_time1);
    MPI_Reduce(&partial_sum_trans_par, &sum_trans_par, 1, MPI_DOUBLE, MPI_SUM,
                0, MPI_COMM_WORLD);

// RESULTS PRINTING

    if(rank == 0) {
        printf("Attempt number PARALLEL: %d --------\n", i+1);
        printf("WTime in Parallel: %f seconds\n\n", par_trans_time2-par_trans_time1);
    }

}

    if(rank == 0) {
        avg_trans_seq = sum_trans_seq / 50;
        avg_trans_par = sum_trans_par / 50;

        printf("AVERAGE SEQ WTime is: %f seconds\n", avg_trans_seq);
        printf("AVERAGE PAR WTime is: %f seconds\n", avg_trans_par);
    }

// DEALLOCATION

    if(rank == 0) {
        free_matrix(matrix, N);
        free_matrix(matrix_transposed, N);
    }

    MPI_Finalize();
    
    return 0;

}