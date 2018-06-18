// assume B is transposed for now
// Arithmetic intensity of 2


__attribute__((max_global_work_dim(0)))
__kernel void matrix_mult( 
	__global float* restrict A, __global float* restrict B, __global float* restrict C, int rowsC, int colsC, int k_size )
{  
    for(int r = 0 ; r < rowsC; r++){
        for(int c = 0 ; c < colsC; c++){
            float res = 0;
            for(int k = 0 ; k < k_size ; k++){
                res+=A[r * k_size + k] * B[c * k_size + r]; //A[r][k] * B[c][k]
            }
            C[r * colsC + c] = res;
        }
    }
}


