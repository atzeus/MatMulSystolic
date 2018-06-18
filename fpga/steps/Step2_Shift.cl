// assume B is transposed for now
// Arithmetic intensity of 2

// Step 1: Increase AI by using local memory loop tiling, why does compiler not report inner loop?

#define BLOCK_SIZE_ROWS 32
#define BLOCK_SIZE_COLS 32
#define K_SIZE 4000

__attribute__((max_global_work_dim(0)))
__kernel void matrix_mult( 
	__global float* restrict A, __global float* restrict B, __global float* restrict C, int rowsC, int colsC, int k_size )
{  
  #pragma loop_coalesce 2
  for(int r = 0 ; r < rowsC; r+=BLOCK_SIZE_ROWS){
    for(int c = 0 ; c < colsC; c+=BLOCK_SIZE_COLS){
      float res[BLOCK_SIZE_ROWS][BLOCK_SIZE_COLS];

      #pragma unroll
      for(int rb = 0 ; rb < BLOCK_SIZE_ROWS; rb++){
        #pragma unroll
        for(int rc = 0 ; rc < BLOCK_SIZE_COLS; rc++){
          res[rb][rc] = 0;
        }
      }

      for(int k = 0 ; k < k_size ; k++){
        float ALocal[BLOCK_SIZE_ROWS];
        float BLocal[BLOCK_SIZE_COLS];
        #pragma unroll 8
        for(int rb = 0 ; rb < BLOCK_SIZE_ROWS; rb++){
          ALocal[rb] = A[(r + rb) * k_size + k] ;
        }
        #pragma unroll 8
        for(int rc = 0 ; rc < BLOCK_SIZE_COLS; rc++){
          BLocal[rc] = B[(c + rc) * k_size + k];
        }

        #pragma loop_coalesce 2
        #pragma unroll 32
        for(int rb = 0 ; rb < BLOCK_SIZE_ROWS; rb++){
          for(int rc = 0 ; rc < BLOCK_SIZE_COLS; rc++){
            res[rb][rc]+=ALocal[rb] * BLocal[rc];
          }
        }
      }
      #pragma unroll 1
      for(int rb = 0 ; rb < BLOCK_SIZE_ROWS; rb++){
        #pragma unroll 16
        for(int rc = 0 ; rc < BLOCK_SIZE_COLS; rc++){
          C[(r + rb) * colsC + (c + rc)] = res[rb][rc];
        }
      }

    }
  }
}


