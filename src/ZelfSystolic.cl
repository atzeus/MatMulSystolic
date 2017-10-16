#include "host/inc/PE_systolic_array_10x16_dot8_interleaving_1024_large_mat.h"
#include "host/inc/matrixMul_10x16_large.h"
#include "host/inc/PE_systolic_array_generic_matrix_blocking_params.h"



#define VECTOR_FLOAT4_ZERO   (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO   (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)

#if DOT_PROD_VECTOR_SIZE==4
typedef float4 vec_float_t;
#define VECTOR_ZERO VECTOR_FLOAT4_ZERO
#elif DOT_PROD_VECTOR_SIZE==8
typedef float8 vec_float_t;
#define VECTOR_ZERO VECTOR_FLOAT8_ZERO
#elif DOT_PROD_VECTOR_SIZE==16
typedef float16 vec_float_t;
#define VECTOR_ZERO VECTOR_FLOAT16_ZERO
#else
#error Unsupported DOT_PROD_VECTOR_SIZE
#endif


#ifndef EMULATOR  // don't use packed in the emulator
__attribute__((packed))
#endif
struct ch_data_a_struct {
    vec_float_t data;
    bool new_row_col_pair;
};

#ifndef EMULATOR  // don't use packed in the emulator
__attribute__((packed))
#endif
struct ch_data_b_struct {
    vec_float_t data;
};

#ifndef EMULATOR  // don't use packed in the emulator
__attribute__((packed))
#endif
struct ch_data_c_struct {
    float data;
};

#ifndef EMULATOR
__attribute__((packed))
#endif
struct custom_float_array { 
	float vals[SYS_ARRAY_NUM_COLS];
};

#define INTERLEAVED 1
#define MATRIX_A_BLOCK_HEIGHT  (INTERLEAVED * SYS_ARRAY_NUM_ROWS)
#define MATRIX_B_BLOCK_WIDTH   (INTERLEAVED * SYS_ARRAY_NUM_COLS)

#define NR_INTERLEAVED (INTERLEAVED * INTERLEAVED)

channel struct ch_data_a_struct  ch_data_a_border [SYS_ARRAY_NUM_ROWS]                       __attribute__((depth(0)));
channel struct ch_data_a_struct  ch_data_a        [SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS-1] __attribute__((depth(0)));

channel vec_float_t              ch_data_b        [SYS_ARRAY_NUM_ROWS-1][SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));
channel vec_float_t              ch_data_b_border                       [SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));

channel float                    ch_data_c        [SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS]   __attribute__((depth(0)));

channel float                    ch_drain_c       [SYS_ARRAY_NUM_ROWS-1][SYS_ARRAY_NUM_COLS] __attribute__((depth(NR_INTERLEAVED)));
channel float                    ch_drain_c_border                      [SYS_ARRAY_NUM_COLS] __attribute__((depth(NR_INTERLEAVED)));

channel struct ch_data_a_struct  row_feed_chain        [SYS_ARRAY_NUM_ROWS-1] __attribute__((depth(0)));
channel struct ch_data_a_struct  row_feed_chain_border                        __attribute__((depth(0)));
channel struct ch_data_a_struct  row_feed_to_buf [SYS_ARRAY_NUM_ROWS]         __attribute__((depth(0)));

channel vec_float_t  col_feed_chain        [SYS_ARRAY_NUM_COLS-1] __attribute__((depth(0)));
channel vec_float_t  col_feed_chain_border                        __attribute__((depth(0)));
channel vec_float_t  col_feed_to_buf [SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));

channel struct custom_float_array  col_c_chain          [SYS_ARRAY_NUM_COLS-1] __attribute__((depth(0)));
channel struct custom_float_array  col_c_chain_border                          __attribute__((depth(0)));

channel struct custom_float_array ch_drain_chain[SYS_ARRAY_NUM_COLS];

#define DRAIN_C_LAST_CHANNEL_DEPTH 1

channel struct custom_float_array ch_drain_C_tree_to_mem __attribute__((depth(DRAIN_C_LAST_CHANNEL_DEPTH)));


__attribute__((max_global_work_dim(0)))
__kernel void load_mat_A_and_forward( __global vec_float_t* restrict A,  unsigned int mat_a_num_vectors_in_row, unsigned int mat_a_num_blocks_in_col, unsigned int mat_b_num_blocks_in_row)
{ 

  bool first = true;
  for (int yblock = 0; yblock < mat_a_num_blocks_in_col; yblock++) {
      for (int reuse = 0; reuse < mat_b_num_blocks_in_row; reuse++) {
          for(int x = 0 ; x < mat_a_num_vectors_in_row ; x++){
              for(int y = 0 ; y < MATRIX_A_BLOCK_HEIGHT; y++){
                  int index = (yblock * MATRIX_A_BLOCK_HEIGHT + y) * mat_a_num_vectors_in_row + x ;
                  struct ch_data_a_struct write;
									write.data = A[index];
									write.new_row_col_pair = !first && x == 0;
                  write_channel_intel(row_feed_chain_border, write);
              }
              first = false;
          }
      }
  }
  // flush last block
  for(int row = 0 ; row <  MATRIX_A_BLOCK_HEIGHT ; row++) {
      struct ch_data_a_struct write;
			write.data = VECTOR_ZERO;
			write.new_row_col_pair = true;
      write_channel_intel(row_feed_chain_border,write );
  }
  // Buffer will not flush without new elements
  for(int row = 0 ; row < MATRIX_A_BLOCK_HEIGHT ; row++) {
      struct ch_data_a_struct write;
			write.data = VECTOR_ZERO;
			write.new_row_col_pair = false;
      write_channel_intel(row_feed_chain_border,write );
  }

}


// input is transposed and blocked
__attribute__((max_global_work_dim(0)))
__kernel void load_mat_B_and_forward( __global vec_float_t* restrict B, unsigned int mat_a_num_vectors_in_row, unsigned int mat_a_num_blocks_in_col, unsigned int mat_b_num_blocks_in_row )
{
    int mat_b_num_vectors_in_col = mat_a_num_vectors_in_row;
	  for (int reuse = 0; reuse < mat_a_num_blocks_in_col; reuse++) {
        for (int xblock = 0; xblock < mat_b_num_blocks_in_row; xblock++) {
            for (int y = 0; y < mat_a_num_vectors_in_row; y++) {
                for (int x = 0; x < MATRIX_B_BLOCK_WIDTH; x++) {
                    int index = (xblock * MATRIX_B_BLOCK_WIDTH + x) * mat_b_num_vectors_in_col + y;
                    write_channel_intel(col_feed_chain_border, B[index]);
                }
            }
        }
    }
    // flush last block
    for(int row = 0 ; row < MATRIX_B_BLOCK_WIDTH *2   ; row++) {
        write_channel_intel(col_feed_chain_border,VECTOR_ZERO );
    }
}




// instantiate matrix A feeders
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(SYS_ARRAY_NUM_ROWS)))
__kernel void feed_mat_A_kernel()
{
	const int row = get_compute_id(0);
	const int freq = (SYS_ARRAY_NUM_ROWS - 1) - row;
  int count = freq;
  while(true) {
      struct ch_data_a_struct read;
      if(row == 0 ){
          read = read_channel_intel(row_feed_chain_border);
      } else {
          read = read_channel_intel(row_feed_chain[row-1]);
      }
      if(count == 0) {
          write_channel_intel(row_feed_to_buf[row], read);
          count = freq;
      } else {
          write_channel_intel(row_feed_chain[row+1], read);
          count--;
      }
  }
}

// instantiate matrix B feeders
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(SYS_ARRAY_NUM_COLS)))
__kernel void feed_mat_B_kernel()
{
	const int col = get_compute_id(0);
	const int freq = (SYS_ARRAY_NUM_COLS - 1) - col;
  int count = freq;
  while(true) {
      vec_float_t read;
			if(col == 0) {
          read = read_channel_intel(col_feed_chain_border);
      } else {
          read = read_channel_intel(col_feed_chain[col-1]);
      } 
      if(count == 0) {
          write_channel_intel(col_feed_to_buf[col], read);
          count = freq;
      } else {
          write_channel_intel(col_feed_chain[col+1], read);
          count--;
      }
  }
}


// instantiate matrix A buffers
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(SYS_ARRAY_NUM_ROWS)))
__kernel void buf_mat_a_kernel()
{
	const int row = get_compute_id(0);
  struct ch_data_a_struct buf[2][INTERLEAVED];

	#pragma unroll
	for(int b = 0 ; b < 2 ; b++){
		#pragma unroll
		for(int r = 0 ; r < INTERLEAVED ; r++) {
			buf[b][r].data = VECTOR_ZERO;
			buf[b][r].new_row_col_pair = false;
		}
	}

	// We obtain a new value every COLS_INTERLEAVED steps
  // Each value is also reused COLS_INTERLEAVED steps
  // meaning new values come in at exactly the right rate
  int reuse_cnt = 0;
  int buf_to_read = 0;
  int row_to_read_write = 0;
  while (true) {
      write_channel_intel(ch_data_a_border[row],buf[buf_to_read][row_to_read_write] );
      if(reuse_cnt == INTERLEAVED - 1) {
          reuse_cnt = 0;
          // load once every COLS_INTERLEAVED steps
          struct ch_data_a_struct  read = read_channel_intel(row_feed_to_buf[row]);
          int buf_to_write = 1 - buf_to_read;
          buf[buf_to_write][row_to_read_write] = read;

          if(row_to_read_write == INTERLEAVED - 1){
              buf_to_read = 1 - buf_to_read;
              row_to_read_write = 0;
          } else {
              row_to_read_write = row_to_read_write + 1;
          }
      } else {
          reuse_cnt = reuse_cnt + 1;
      }
	}
}




// instantiate matrix B buffers
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(SYS_ARRAY_NUM_COLS)))
__kernel void buf_mat_b_kernel()
{
	const int col = get_compute_id(0);
  vec_float_t buf[2][INTERLEAVED];

	#pragma unroll
	for(int b = 0 ; b < 2 ; b++){
		#pragma unroll
		for(int r = 0 ; r < INTERLEAVED ; r++) {
			buf[b][r] = VECTOR_ZERO;
		}
	}

	// We obtain a new value every ROWS_INTERLEAVED steps
  // Each value is also reused ROWS_INTERLEAVED steps
  // meaning new values come in at exactly the right rate (assuming ROWS_INTERLEAVED = COLS_INTERLEAVED)
  int reuse_cnt_col_to_write = 0;
  int buf_to_read = 0;
  int col_to_read = 0;
  while (true) {
      write_channel_intel(ch_data_b_border[col],buf[buf_to_read][col_to_read]);
      if(col_to_read == INTERLEAVED - 1){
          col_to_read = 0;
          // load once every COLS_INTERLEAVED steps
          vec_float_t read = read_channel_intel(col_feed_to_buf[col]);
          int buf_to_write = 1 - buf_to_read;
          buf[buf_to_write][reuse_cnt_col_to_write] = read;

          if(reuse_cnt_col_to_write == INTERLEAVED - 1){
              buf_to_read =  1- buf_to_read;
              reuse_cnt_col_to_write = 0;
          } else {
              reuse_cnt_col_to_write++;
          }
      } else {
          col_to_read++;
      }

  }
}

	
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(SYS_ARRAY_NUM_ROWS,SYS_ARRAY_NUM_COLS)))
__kernel void PE_kernel()
{
		const int row = get_compute_id(0);
		const int col = get_compute_id(1);
    float interleave_shift[NR_INTERLEAVED];
    
    #pragma unroll
    for (int i=0; i < NR_INTERLEAVED  ; i++) {
        interleave_shift[i] = 0.0f;
    }


    while(true){
      struct ch_data_a_struct read_A;  
		 	if(col == 0) {
      	read_A = read_channel_intel(ch_data_a_border[row]);
      } else {
      	read_A = read_channel_intel(ch_data_a[row][col-1]);
      }

      vec_float_t a_data = read_A.data;
      bool new_col_row_pair = read_A.new_row_col_pair;
      if (col < (SYS_ARRAY_NUM_COLS-1)) {
          write_channel_intel(ch_data_a[row][col+1], read_A);
      }
      vec_float_t b_data;
      if(row == 0){
      	b_data = read_channel_intel(ch_data_b_border[col]);
      } else {
      	b_data = read_channel_intel(ch_data_b[row-1][col]);
      }
      if (row < (SYS_ARRAY_NUM_ROWS-1)) {
          write_channel_intel(ch_data_b[row+1][col], b_data);
      }
      if(new_col_row_pair) {
          write_channel_intel(ch_data_c[row][col],interleave_shift[NR_INTERLEAVED-1]);
      }

      float sum = 0;
      for(int d=0; d < DOT_PROD_VECTOR_SIZE; ++d) {
          sum += a_data[d] * b_data[d];
      }

      float accum = sum + (new_col_row_pair ? 0.0f : interleave_shift[NR_INTERLEAVED-1]);
      for (int i = NR_INTERLEAVED-1; i >= 1; i--) {
          interleave_shift[i] = interleave_shift[i - 1];
      }
      interleave_shift[0] = accum;
  }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(SYS_ARRAY_NUM_ROWS,SYS_ARRAY_NUM_COLS)))
__kernel void drain_C()
{
	const int row = get_compute_id(0);
	const int col = get_compute_id(1);
	int i = 0;
  int interleaved = 0;
  
	while (true) {
      float read;
      if (i == 0){
          read = read_channel_intel(ch_data_c[row][col]);
      } else {
          read = read_channel_intel(ch_drain_c[row-1][col]);
      }
  		if(row == SYS_ARRAY_NUM_ROWS -1){
          write_channel_intel(ch_drain_c_border[col], read);
      } else {
          write_channel_intel(ch_drain_c[row][col], read);
      }
      if(interleaved == INTERLEAVED - 1){
          interleaved = 0;
          if(i == row ){
              i = 0;
          } else {
              i = i + 1;
          }
      } else {
          interleaved++;
     }

  }
}






__attribute__((autorun))
__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(SYS_ARRAY_NUM_COLS)))
__kernel void drain_C_chain_node_kernel() {
	unsigned col = get_compute_id(0);


	while(true){
		  float in = read_channel_intel(ch_drain_c[SYS_ARRAY_NUM_ROWS - 1][col]);
		  struct custom_float_array  prev_node_data_in;
		  if(col != SYS_ARRAY_NUM_COLS - 1) {
		      prev_node_data_in = read_channel_intel(col_c_chain[col +1]);
		  }

		  struct custom_float_array write;

		  for (int i = 0; i < SYS_ARRAY_NUM_COLS - 1; i++) {
		      write.vals[i] = prev_node_data_in.vals[i+1];
		  }

		  write.vals[SYS_ARRAY_NUM_COLS-1] = in;
		  if(col == 0){
          write_channel_intel(col_c_chain_border, write);
      } else {
          write_channel_intel(col_c_chain[col], write);
      }
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void drain_C_write_tree_root_to_mem_kernel(__global struct custom_float_array* restrict C, int nrXBlocks, int nrYBlocks) {
	for(int yblock = 0 ; yblock < nrYBlocks ; yblock++){
      for(int xblock = 0; xblock < nrXBlocks ; xblock++){
          for(int ylocal = 0 ; ylocal < MATRIX_A_BLOCK_HEIGHT ; ylocal++) {
              for (int xlocal = 0; xlocal < INTERLEAVED; xlocal++) {
                  int index = ((yblock * MATRIX_A_BLOCK_HEIGHT + ylocal) * nrXBlocks + xblock) * INTERLEAVED  + xlocal;
                  struct custom_float_array dataIn = read_channel_intel(col_c_chain_border);
                  C[index] = dataIn;
              }
          }
      }
  }
}

    
