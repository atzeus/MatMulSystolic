#include "host/inc/PE_systolic_array_10x16_dot8_interleaving_1024_large_mat.h"
#include "host/inc/matrixMul_10x16_large.h"
#include "host/inc/PE_systolic_array_generic_matrix_blocking_params.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

#define VECTOR_FLOAT1_ZERO   0.0f
#define VECTOR_FLOAT4_ZERO   (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO   (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)
#if DOT_PROD_VECTOR_SIZE==1
typedef float vec_float_t;
#define VECTOR_ZERO VECTOR_FLOAT1_ZERO
#elif DOT_PROD_VECTOR_SIZE==4
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

//#define INTERLEAVED 32
#define MATRIX_A_BLOCK_HEIGHT  (INTERLEAVED * SYS_ARRAY_NUM_ROWS)
#define MATRIX_B_BLOCK_WIDTH   (INTERLEAVED * SYS_ARRAY_NUM_COLS)

#define INTERLEAVED_SQUARED (INTERLEAVED * INTERLEAVED)


#ifndef EMULATOR  // don't use packed in the emulator
__attribute__((packed))
#endif
struct ch_data_a_struct {
    vec_float_t data;
    bool last;
};


#ifndef EMULATOR  // don't use packed in the emulator
__attribute__((packed))
#endif
struct ch_data_b_struct {
    vec_float_t data;
};


#ifndef EMULATOR
__attribute__((packed))
#endif
struct custom_float_array { 
	float vals[SYS_ARRAY_NUM_COLS];
};


/* This implements a systolic array matrix multiplication of matrix A with 
   matrix B, consisting of n x m processing elements (PE).

  The systolic array itself computes a block of the output matrix of size 
  (n * i * v) x (m * i * v) where i is the interleaving factor (constant INTERLEAVED)
  and v is the vector size (constant DOT_PROD_VECTOR_SIZE) (more on this later).

  There are separate loader kernels for A and B, which reads the data 
  necessary for a computing a block 
  from memory. After computation, the result is written 
  by a writer kernel. This process is repeated for each block in the output matrix
  thereby eventually computing the entire matrix. The necessary data 
  for a block is a contiguous group of (n*i*v) rows in matrix A and a contiguous group
  of (m * i* v) columns in matrix B.  

The flow data in the design is shown in diagram.png (n = m = 2)
 

The implementation consist of the following elements

Load A :

    kernel name : load_mat_A_and_forward,

    This loads matrix A from memory and feeds it to the first A feeder.


    The rows of matrix A are fed one vector (= 1,2,4,8 or 16 values) at a time. 
    Each vector contains multiple (contiguous) values from the same row. For
    instance if A(i,j) is the value of matrix A at row i and column j, then
    a 4 value vector might consist of <A(1,4),A(1,5),A(1,6),A(1,7)>.

    The process of feeding vectors of A proceeds from top -to -bottom first, then
    left-to-right, enumerating vectors in a (contiguous) block of columns with 
    the group of rows before moving to the next block of columns. For example, 
    if the vector size is 2 and the block size 3, then the values fed by the A 
    loader are:  <A(0,0),A(0,1)> , <A(1,0),A(1,1)>, <A(2,0),A(2,1)> ,
                 <A(0,2),A(0,3)>  ...

    The A loader also feeds an extra bit with each vector which indicates
    if data from the PEs needs to be flushed (written to memory). 

    This boolean is true whenever the column is 0, except for the first output
    block (no results to flush yet). 
    

Load B:

    kernel name:  load_mat_B_and_forward

    The load of B works analogously to the load of A, but transposed. Vectors consist of multiple
    values from the same column instead of row etc. 

    The B loader does not feed an extra bit with each vector which which indicates
    if data from the PEs needs to be flushed, as this is already present in the
    vectors from A.


Feed A:
    
    kernel names : feed_mat_A_kernel_1 .. feed_mat_a_kernel_n,
                  
    Together, these kernels get data from the A loader and feed that data, 
    without modification, in round robin fashion the buffers of A. The first received
    data from the loader is fed to Buf A 0 the next to Buf A 1 and so forth.
    Hence the data rate on the channels col_feed_to_buf is 1/n of the data rate of 
    the col_feed_chain 0 channel

Feed B: 
    kernel names : feed_mat_B_kernel_1 .. feed_mat_B_kernel_m,
                  
   The same as Feed A, but for B.

Buf A:
    
    kernel names : buf_mat_a_kernel_1 .. buf_mat_a_kernel_n
                  
   In the case of matrix a, these buffer kernels do not feature a buffer,
   but repeat whatever they receive i times.  Hence the rate of out coming
   messages is i times the rate of incoming messages.

Buf B:
    
    kernel names : buf_mat_b_kernel_1 .. buf_mat_b_kernel_n

    These kernels each have a buffer of size i. The contents of this
    buffer are fed in order to the PEs i times before switching to 
    the next INTERLEAVED values. To overlap sending and receiving messages,
    a double-buffer strategy is used. Note that because each incoming value
    is output i times, the rate of out coming messages is i times the rate of
    incoming messages.

PE:
    kernel names: PE_kernel_1,1 .. PE_kernel_n,1  ... PE_kernel PE_kernel_n,m

    The processing element perform the actual computation. On each time step,
    PE_i,j obtains a vector from matrix A from ch_data_a_i,j which also 
    forwards to the PE to the right, and a vector from matrix B from 
    ch_data_b_i,j which it forwards to the PE below. 
 
    The dot product of the two vectors is computed and added to a stored sum.

    Each PE computes i * i (= INTERLEAVED_SQUARED) results of an output block. 
    The reason for this is twofold: 
      * Each value from memory is now reused an INTERLEAVED times more than
        without interleaving.
      * Latency hiding: 
        The result of the fused multiply-adds used in computing the dotproduct
        are not immediately available for use in the next cycle. By iterating 
        over the i * i output values (stored in a shift-register)  in round 
        robin fashion, we do not need the result until i * i cycles later.

    Within an output block, the following results are computed by the PE at 
    index x (column) y (row) (the shift register contains them in this order): 

    [ (x + ix * m , y + iy * n) | ix <- [0..i-1] , iy <- [0..i-1]]

    Note that this is also dictates how the Bufs in A/B work.

    When the extra bit added to the vector from matrix A indicates that a 
    result should be flushed, the value currently at the head of the shift 
    register is output on ch_data_c and a the current sum is set to 0.

    To make sure there is always space to write drain_c, this is the only channel
    with a buffer, namely of size i * i.

Write C: 
    kernel names : drain_C_write_tree_root_to_mem_kernel

    Writes back the results of matrix C to memory. Within each block, 
    each time it receives a message, it receives the next m (=number of columns) 
    values of the output block which it then stores.

Drain_chain C:

    kernel names: drain_C_chain_node_kernel_n

   Each value obtains the values of the right drain_chains and a new single value
   from the PE above, bundles them and passes them on to the left. 
   In this way, at the end on the left m simultaneous values come out per message.

Drain C:

    kernel names: drain_C_1,1 ... drain_C_n,m

    The drain kernels pass on information from above and from the associated PE.
    By the order in which they alternate to pass on info from the PE or the 
    drains above, they make sure that the results of the computation are drained
    in the right order such that the value which end up at drain_chain together
    for m contiguous values from a row in the output block.


Optimizations to try out: 

Writes/reads to channel in 1 places
Replace buffer in Buf B with shift-registers.
Replace for-loops with while loops.

       
*/


channel struct ch_data_a_struct  ch_data_a        
	[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));
channel struct ch_data_b_struct		 ch_data_b       
	[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));
channel float                    ch_data_c        
	[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS]  __attribute__((depth(INTERLEAVED_SQUARED)));

channel float                    ch_drain_c       
	[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));
channel struct ch_data_a_struct  row_feed_chain        
	[SYS_ARRAY_NUM_ROWS] __attribute__((depth(0)));
channel struct ch_data_a_struct  row_feed_to_buf 
	[SYS_ARRAY_NUM_ROWS]         __attribute__((depth(0)));

channel struct ch_data_b_struct  	 col_feed_chain        
	[SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));
channel struct ch_data_b_struct	     col_feed_to_buf 
	[SYS_ARRAY_NUM_COLS] __attribute__((depth(INTERLEAVED)));

channel struct custom_float_array  col_c_chain          
	[SYS_ARRAY_NUM_COLS] __attribute__((depth(0)));






/* Explanation of names:

nrXBlocks : 
	the width of the output matrix, 
    as the number of blocks of width MATRIX_B_BLOCK_WIDTH 
	        
    The width of matrix b is nrXBlocks * MATRIX_B_BLOCK_WIDTH
        
nrYBlocks : 
	the height of the output matrix 	    
	as the number of block of height MATRIX_A_HEIGHT
			
	The width of matrix a is nrYBlocks * MATRIX_A_BLOCK_WIDTH 
			

dotProdVecLength : 
	The (width of matrix a = height of matrix b) divided by 
	DOT_PROD_VECTOR_SIZE. Each element in the output matrix is the sum of a 
	number of dotProdVecLength  dot products between vectors of size 
	DOT_PROD_VECTOR_SIZE
*/

void feedRowBlock(__global  vec_float_t* restrict A, int rowBlock,
	int dotProdVecLength);

__attribute__((max_global_work_dim(0)))
__kernel void load_mat_A_and_forward( 
	__global vec_float_t* restrict A, unsigned int nrXBlocks, unsigned int nrYBlocks, 
					unsigned int dotProdVecLength)
{ 
  for(int rowBlock = 0 ; rowBlock < nrYBlocks ; rowBlock++){
  	for(int reuse = 0 ; reuse < nrXBlocks ; reuse++){
  		feedRowBlock(A,rowBlock, dotProdVecLength);
    }
  }
}

void feedRowBlock(__global  vec_float_t* restrict A, int rowBlock,
	int dotProdVecLength){
    const int lastCol = dotProdVecLength - 1;
    for(int col = 0 ; col < dotProdVecLength; col++){
        for(int row = 0 ; row < MATRIX_A_BLOCK_HEIGHT ; row++){
        	int index =  (rowBlock * MATRIX_A_BLOCK_HEIGHT + row) 
        					* dotProdVecLength + col;
        	struct  ch_data_a_struct write;
			write.data = A[index];
			// the "last" boolean indicates if an old result 
			// should be flushed
			// this is the case if on the last column
			write.last = col == lastCol;
		    write_channel_intel(row_feed_chain[0],write);				
        }
    }
}

void feedColumnBlock(__global vec_float_t* restrict B, unsigned int colBlock, 
    unsigned int dotProdVecLength);

// input is transposed 
__attribute__((max_global_work_dim(0)))
__kernel void load_mat_B_and_forward( __global vec_float_t* restrict B, 
	unsigned int  nrXBlocks, unsigned int nrYBlocks, 
	unsigned int dotProdVecLength)
{
	for(int reuse = 0 ; reuse < nrYBlocks; reuse++){
		for(int colBlock = 0 ; colBlock < nrXBlocks ; colBlock++){
		    feedColumnBlock(B,colBlock, dotProdVecLength);
		}
	}
}

void feedColumnBlock(__global vec_float_t* restrict B, unsigned int colBlock, 
    unsigned int dotProdVecLength){
    for(int row = 0 ; row < dotProdVecLength ; row++){
        for(int col = 0 ; col < MATRIX_B_BLOCK_WIDTH  ; col++){
            int index = (colBlock * MATRIX_B_BLOCK_WIDTH + col) * 
                             dotProdVecLength + row;
            struct  ch_data_b_struct write;
            write.data = B[index];
            write_channel_intel(col_feed_chain[0],write);
        }
    }
}

// The feeders obtain data from the loader and distribute the data
// round robin fashion over the buffers

// instantiate matrix A feeders
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(SYS_ARRAY_NUM_ROWS)))
__kernel void feed_mat_A_kernel()
{
	const int row = get_compute_id(0);
    const int nrFeedersBelow = (SYS_ARRAY_NUM_ROWS - 1) - row;
    char i = 0;
	while (true) {
		struct  ch_data_a_struct read = read_channel_intel(row_feed_chain[row]);
        if( i == 0) {
            write_channel_intel(row_feed_to_buf[row], read);
            i = nrFeedersBelow;
        } else {
		    write_channel_intel(row_feed_chain[row+1], read);
            i--;
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
	const int nrFeedersRight = (SYS_ARRAY_NUM_COLS - 1) - col;
    char i = 0;
 	while(true) {
        struct  ch_data_b_struct read = read_channel_intel(col_feed_chain[col]);
        if(i == 0){
		    write_channel_intel(col_feed_to_buf[col], read);
            i = nrFeedersRight;
        } else {
            write_channel_intel(col_feed_chain[col+1], read);
            i--;
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
	// Matrix A buffers just repeat everything they
	// get INTERLEAVED times
    struct  ch_data_a_struct feed;
    char i = 0;
	while(true){
		if(i == 0){
            feed = read_channel_intel(row_feed_to_buf[row]);
        } 
		write_channel_intel(ch_data_a[row][0],feed);
        if(i == INTERLEAVED - 1){
            i = 0;
        } else {
            i++;
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
	struct  ch_data_b_struct buf[INTERLEAVED];
    #pragma unroll
	for(int i = 0 ; i < INTERLEAVED ; i++){
		buf[i].data = 0.0f;
	}
	int it = 0;
	while(true){
        struct  ch_data_b_struct cur;
        if(it < INTERLEAVED){
            cur = read_channel_intel(col_feed_to_buf[col]);
        } else {
            cur = buf[INTERLEAVED - 1];
        }
        write_channel_intel(ch_data_b[0][col],cur);
        #pragma unroll
		for(int i = INTERLEAVED -1 ; i >= 1 ;i--)
			buf[i] = buf[i-1];
        buf[0] = cur;
       
        if(it == INTERLEAVED_SQUARED - 1){
            it = 0;
        } else {
            it++;
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
    float interleave_shift[INTERLEAVED_SQUARED];
    
    #pragma unroll
    for (int i=0; i < INTERLEAVED_SQUARED  ; i++) {
        interleave_shift[i] = 0.0f;
    }

    int p = 0 ;
	while(true){
		struct  ch_data_a_struct read_A =
			 read_channel_intel(ch_data_a[row][col]);
		if (col < (SYS_ARRAY_NUM_COLS-1))
			 write_channel_intel(ch_data_a[row][col+1], read_A);

		struct  ch_data_b_struct b_data = 
			read_channel_intel(ch_data_b[row][col]);
		if (row < (SYS_ARRAY_NUM_ROWS-1)) 
			write_channel_intel(ch_data_b[row+1][col], b_data);

		float sum;
		
	    sum = interleave_shift[INTERLEAVED_SQUARED -1];
		

		#pragma unroll
		for(int d=0; d < DOT_PROD_VECTOR_SIZE; ++d) 
			sum += read_A.data[d] * b_data[d];
	
		#pragma unroll
		for(int i = INTERLEAVED_SQUARED -1 ; i >= 1 ;i--)
			interleave_shift[i] = interleave_shift[i-1];

        if(read_A.last) {
			write_channel_intel(ch_data_c[row][col],
				sum);
			interleave_shift[0] = 0.0f;
		} else {
		    interleave_shift[0] = sum;
        }
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
 
	while (true) {
        float read ;
        if( i < INTERLEAVED * row ) {
            read = read_channel_intel(ch_drain_c[row - 1][col]);
        } else {
              read = read_channel_intel(ch_data_c[row][col]);
        }
        write_channel_intel(ch_drain_c[row][col], read);
        if(i == INTERLEAVED * (row + 1) - 1){
            i = 0;
        } else {
            i++;
        }
	}
	

 
}


__attribute__((autorun))
__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(SYS_ARRAY_NUM_COLS)))
__kernel void drain_C_chain_node_kernel() {
	unsigned col = get_compute_id(0);

	while(true){
		float in = read_channel_intel(ch_drain_c[SYS_ARRAY_NUM_ROWS-1][col]);
		struct custom_float_array prev_node_data_in;
		if(col != SYS_ARRAY_NUM_COLS - 1)
		    prev_node_data_in = read_channel_intel(col_c_chain[col + 1]);

		struct custom_float_array write;
        #pragma unroll
		for (int i = col + 1; i < SYS_ARRAY_NUM_COLS; i++)
		    write.vals[i] = prev_node_data_in.vals[i];

		write.vals[col] = in;
		write_channel_intel(col_c_chain[col],write);


	}
}

__attribute__((max_global_work_dim(0)))
__kernel void drain_C_write_tree_root_to_mem_kernel(__global struct custom_float_array* restrict C, int nrXBlocks, int nrYBlocks) {
	 int num_vec_per_row = INTERLEAVED * nrXBlocks;
	for(int yblock = 0 ; yblock < nrYBlocks ; yblock++){
		for(int xblock = 0; xblock < nrXBlocks ; xblock++){
		    for(int ylocal = 0 ; ylocal < MATRIX_A_BLOCK_HEIGHT ; ylocal++) {
		        for (int xlocal = 0; xlocal < INTERLEAVED; xlocal++) {
		            int index = ((yblock * MATRIX_A_BLOCK_HEIGHT + ylocal) * num_vec_per_row) + (xblock * INTERLEAVED) + xlocal;
                    printf("got data xblock %d yblock %d x %d y %d \n",  xblock, yblock, xlocal,ylocal);
		            struct custom_float_array dataIn = read_channel_intel(col_c_chain[0]);
		            C[index] = dataIn;
                    
		        }
		    }
		}
	}
    printf("Done!\n");
}
   
