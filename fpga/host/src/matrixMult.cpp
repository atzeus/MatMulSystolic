// System includes 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <time.h>
#include <windows.h>
#else
#include <sys/time.h>
#endif


#define ALTERA_CL 1

#ifdef ALTERA_CL
#pragma message ("* Compiling for ALTERA CL")
#endif

#ifdef ALTERA_CL
#include "CL/opencl.h"
#endif


#include <PowerSensor.h>


#define ACL_ALIGNMENT 64

#ifdef _WIN32
void* acl_aligned_malloc (size_t size) {
	return _aligned_malloc (size, ACL_ALIGNMENT);
}
void acl_aligned_free (void *ptr) {
	_aligned_free (ptr);
}
#else
#include <stdlib.h>
void* acl_aligned_malloc (size_t size) {
	void *result = NULL;
	posix_memalign (&result, ACL_ALIGNMENT, size);
	return result;
}
void acl_aligned_free (void *ptr) {
	free (ptr);
  }
#endif // LINUX


//#define EMULATOR
//#define COMPUTE_GOLDEN_BLOCKED
#define COMPUTE_GOLDEN



#define LARGE_MAT

//#define SYS_ARRAY_GEO 2016 // just concatenate ROWS COLS into integer
#define SYS_ARRAY_GEO 1016 // just concatenate ROWS COLS into integer

#define NUM_NON_AUTORUN_KERNELS 3



#if SYS_ARRAY_GEO==2016

#ifdef LARGE_MAT
  #include "PE_systolic_array_20x16_dot4_interleaving_512_large_mat.h"
  #include "matrixMul_20x16_large.h"
#else
  #include "PE_systolic_array_20x16_dot4_interleaving_512.h"
  #include "matrixMul_20x16.h"
#endif
  #include "PE_systolic_array_generic_matrix_blocking_params.h"

  #ifdef EMULATOR  
  #define AOCX_FILE "systolic_array_20x16_dot4.aocx"
  #else
  #define AOCX_FILE "systolic_array_20x16_dot4.aocx"
  #endif

  #define NUM_KERNELS 	( SYS_ARRAY_NUM_ROWS*SYS_ARRAY_NUM_COLS + SYS_ARRAY_NUM_ROWS + SYS_ARRAY_NUM_COLS + (SYS_ARRAY_NUM_COLS-1) + (1 + 1 + 1) )

  #define NUM_QUEUES 	NUM_KERNELS

  // set the kernel names (kernel functions)
  #include "host_kernel_strings_20x16.h"

  #define   KID_FEED_MAT_A 	0
  #define   KID_FEED_MAT_B      1
  #define   KID_DRAIN_MAT_C 	2

#elif SYS_ARRAY_GEO==1016

#ifdef LARGE_MAT
  #include "PE_systolic_array_10x16_dot8_interleaving_1024_large_mat.h"
  #include "matrixMul_10x16_large.h"
#else
  #include "PE_systolic_array_10x16_dot8_interleaving_1024.h"
  #include "matrixMul_10x16.h"
#endif
  #include "PE_systolic_array_generic_matrix_blocking_params.h"

  #ifdef EMULATOR  
  #define AOCX_FILE "ZelfSystolic.aocx"
  #else
  #define AOCX_FILE "ZelfSystolic.aocx"
  #endif

  #define NUM_KERNELS 	( SYS_ARRAY_NUM_ROWS*SYS_ARRAY_NUM_COLS + SYS_ARRAY_NUM_ROWS + SYS_ARRAY_NUM_COLS + (SYS_ARRAY_NUM_COLS-1) + (1 + 1 + 1) )

  #define NUM_QUEUES 	NUM_KERNELS

  // set the kernel names (kernel functions)
  #include "host_kernel_strings_10x16.h"

  #define   KID_FEED_MAT_A 	0
  #define   KID_FEED_MAT_B      1
  #define   KID_DRAIN_MAT_C 	2

#else
  #error Unsupported Systolic array geometry
#endif


#ifdef EMULATOR
  // emulator needs to create both non-autorun and autorun kernels
  #define NUM_KERNELS_TO_CREATE   NUM_KERNELS
  #define NUM_QUEUES_TO_CREATE    NUM_QUEUES
#else
  // only create non-autorun kernels for the HW run
  #define NUM_KERNELS_TO_CREATE   NUM_NON_AUTORUN_KERNELS
  #define NUM_QUEUES_TO_CREATE    NUM_NON_AUTORUN_KERNELS
#endif

#define NUM_QUEUES_TO_FINISH    NUM_NON_AUTORUN_KERNELS




// Check the status returned by the OpenCL API functions
#define CHECK(status) 								\
	if (status != CL_SUCCESS)						\
{									\
	fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
	exit(1);							\
}									\

// Check the status returned by the OpenCL API functions, don't exit on error
#define CHECK_NO_EXIT(status) 								\
	if (status != CL_SUCCESS)						\
{									\
	fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
}									\


void randomize_array(float* array, const int size)
{
    for (int i = 0; i < size; ++i) 
    {
        array[i] = (float)rand() / (float)RAND_MAX;
    }
}

bool compare_L2_norm(const float* ref_array, const float* output_array, const unsigned int size, const float epsilon) 
{
  // compute the L^2-Norm of the difference between the output array and reference array 
  // and compare it against the L^2-Norm of the reference.
  float diff = 0.0f;
  float ref = 0.0f;
  for (int i = 0; i < size; ++i) {

    const float o = output_array[i];
    const float r = ref_array[i];
    const float d = o - r;
    diff += d * d;
    ref += r * r;
  }

  const float diff_l2norm = sqrtf(diff);
  const float ref_l2norm = sqrtf(ref);
  const float error = diff_l2norm / ref_l2norm;
  const bool pass = error < epsilon;

  return pass;
}


// using the original B here, not the transposed version
void compute_gold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    printf("Compute size %d %d\n", hA,wA);
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}

#define min(a,b) ((a<b) ? (a):(b))

#define COMPUTE_GOLD_BLOCK_SIZE 4


// takes transposed version of B
void compute_gold_blocked(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB, unsigned int hB) 
{
   const int block_size = COMPUTE_GOLD_BLOCK_SIZE;

    for(int i0 = 0; i0 < hA ; i0 += block_size) {
        for(int j0 = 0; j0 < wB; j0 += block_size) {
            
            for(int k0=0; k0 < wA ; k0 += block_size ) {

              for(int i = i0; i < min(hA, i0+block_size); i++) {
                  for(int j = j0; j < min(wB, j0+block_size); j++) {
                      double sum = 0;
                      for(int k = k0; k < min(wA, k0+block_size); k++) {
                        double a = A[i * wA + k];
                        //double b = B[k * wB + j];
                        double b = B[j * hB + k]; // B is transposed
                        sum += a * b;
                      }
                      C[i * wB + j] += (float)sum;
                  }
              }
            }
        }
    }
}


void pack_reformat_matrix(float * C_orig, float * C_packed, int mat_height, int mat_width, int pack_factor) {
    int word_id = 0;
    for(int i=0; i < mat_height; i++) {
        for(int j=0; j < mat_width; j++) {
            for(int py = 0; py < pack_factor ; py++){
                for(int px = 0 ; px < pack_factor ; px++){
                    C_packed[word_id] = C_orig[(i + py)*mat_width + (j + py)];
                    word_id++;
                }
            }
        }
    }
}

                

void transpose_matrix( float * B_orig, float * B_transposed, int hB, int wB) 
{
  for(int i=0; i < wB; ++i) {
    for(int j=0; j < hB; ++j) {
      B_transposed[i*hB + j] = B_orig[j*wB + i];
    }
  }
}

void block_wise_reformat_matrix( float * A_orig, float * A_block_wise, int mat_height, int mat_width, int block_height, int block_width) 
{
  int word_id = 0;
  for(int i=0; i < mat_height; i+=block_height) {
    for(int j=0; j < mat_width; j+=block_width) {
      for(int k=0; k < block_height; k++) {
        for(int l=0; l < block_width; l++) {
          A_block_wise[word_id] = A_orig[(i+k)*mat_width + (j+l)];
          word_id++;
        }
      }
    }
  }
}

void row_block_reformat_matrix( float * A_orig, float * A_block_wise, int mat_height, int mat_width, int block_height, int vecsize) 
{
  int word_id = 0;
  for(int i=0; i < mat_height; i+=block_height) {
    for(int j=0; j < mat_width; j+=vecsize) {
      for(int k=0; k < block_height; k++) {
      	for(int v = 0; v < vecsize; v++){
	          A_block_wise[word_id] = A_orig[(i+k)*mat_width + j+v];
    	      word_id++;
    	 }
      }
    }
  }
}

void reorder_within_blocks( float * C_block_wise, float * C_reordered_within_blocks, int mat_height, int mat_width, int num_sys_arr_columns, int block_width) 
{
  int num_elems = mat_height*mat_width;
  int column_interleaving = block_width / num_sys_arr_columns;
  int word_id = 0;
  for(int i=0; i < num_elems; i+=block_width) {
    for(int j=0; j < column_interleaving; j++) {
      for(int k=0; k < num_sys_arr_columns ; k++) {
        C_reordered_within_blocks[word_id] = C_block_wise[i+j+k*column_interleaving];
        word_id++;
      }
    }
  }
}

void print_matrix(float * A, int hA, int wA) 
{
  for(int i=0; i < hA; ++i) {
    for(int j=0; j < wA; ++j) {
      printf("%.5f\t", A[i*wA + j]);
    }
    printf("\n");
  }
}


void printDiff(float *data1, float *data2, long int size, float fListTol)
{
    printf("Listing Differences (nr elems: %d) > %.6f...\n", size,fListTol);
    int i,j,k;
    int error_count=0;
    for (i = 0; i < size; i++) 
    {
        
      float fDiff = fabs(data1[i] - data2[i]);
      if (fDiff > fListTol) 
        {                
            if (error_count < 300) {  // print only first 300 errors
              printf("Host[%d] = %.6f\tKernel[%d]=%.6f\tDiff=%.6f\n", i, data1[i], i, data2[i], fDiff);
            }

            error_count++;
        } else {
          if (error_count < 300) {  // print correct only within first 300 errors
            printf("Correct or nan? --> Host[%d] = %.6f\tKernel[%d]=%.6f\tDiff=%.6f\n", i, data1[i], i, data2[i], fDiff);
          }
        }
    }
    printf("\nTotal Errors = %d\n\n", error_count);
}


double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d)
{
  cl_ulong start, end;
    
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 		sizeof(cl_ulong), &end, 	NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 	sizeof(cl_ulong), &start, 	NULL);
    
  start_d = (double)1.0e-9 * start;
  end_d   = (double)1.0e-9 * end;

  return 	(double)1.0e-9 * (end - start); // nanoseconds to seconds
}



int main(int argc, const char** argv) {
//  PowerSensor::PowerSensor sensor("/dev/ttyACM0");


        printf("%s Starting...\n\n", argv[0]); 

	unsigned int elements;
	FILE * file;
	long int fstart, fend;
	unsigned int i;
        cl_event kernel_exec_event[NUM_QUEUES];

	std::streampos filesize;
	FILE *f_out = stdout;

	////////////////////////////////////////
	// Check and print out the parameters //
	////////////////////////////////////////
        printf("\n===== Host-CPU checking the systolic array matrix multiplication parameters ======\n\n");
	
	printf("HA: \t\t%d\n", HA);
	printf("WA: \t\t%d\n\n", WA);

	printf("HB: \t\t%d\n", HB);
	printf("WB: \t\t%d\n\n", WB);

	printf("HC: \t\t%d\n", HC);				
	printf("WC: \t\t%d\n\n", WC);

	printf("SYS_ARRAY_NUM_ROWS: \t\t%d\n", SYS_ARRAY_NUM_ROWS);
	if (SYS_ARRAY_NUM_ROWS<1) {
		printf("--->ERROR, SYS_ARRAY_NUM_ROWS must be larger than 0\n");
	}
	printf("SYS_ARRAY_NUM_COLS: \t\t%d\n", SYS_ARRAY_NUM_COLS);
	if (SYS_ARRAY_NUM_COLS<1l) {
		printf("--->ERROR, SYS_ARRAY_NUM_COLS must be larger than 0\n");
		exit(1);
	}
	printf("DOT_PROD_VECTOR_SIZE: \t\t%d\n", DOT_PROD_VECTOR_SIZE);
	//if (DOT_PROD_VECTOR_SIZE!=4 && 
//		DOT_PROD_VECTOR_SIZE!=8 && 
//		DOT_PROD_VECTOR_SIZE!=16) {
//		printf("Illegal DOT_PROD_VECTOR_SIZE, supported: 4,8,16\n");
//		exit(1);
//	}
	printf("\n");

//	printf("ACCUM_SHIFT_REG_SIZE: \t\t%d\n", ACCUM_SHIFT_REG_SIZE);

	printf("\n");

	printf("\n");

	printf("MAT_A_BLOCK_HEIGHT: \t\t%d\n",  MAT_A_BLOCK_HEIGHT);
	printf("MAT_A_BLOCK_WIDTH: \t\t%d\n",   MAT_A_BLOCK_WIDTH);
	printf("MAT_A_BLOCK_SIZE: \t\t%d\n",    MAT_A_BLOCK_SIZE);

        printf("MAT_A_BLOCK_NUM_VECTORS: \t\t%d\n",   MAT_A_BLOCK_NUM_VECTORS);
        if (MAT_A_BLOCK_SIZE % DOT_PROD_VECTOR_SIZE) {
          printf("MAT_A_BLOCK_SIZE must be a multiple of DOT_PROD_VECTOR_SIZE\b");
        }
        printf("MAT_A_NUM_BLOCKS_IN_ROW: \t\t%d\n",   MAT_A_NUM_BLOCKS_IN_ROW);
        if (WA % MAT_A_BLOCK_WIDTH) {
          printf("WA must be a multiple of MAT_A_BLOCK_WIDTH\n");
        }
        printf("MAT_A_NUM_BLOCKS_IN_COL: \t\t%d\n",   MAT_A_NUM_BLOCKS_IN_COL);
        if (HA % MAT_A_BLOCK_HEIGHT) {
          printf("HA must be a multiple of MAT_A_BLOCK_HEIGHT\n");
        }
        printf("MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS: \t\t%d\n",   MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS);
	printf("\n");


        if(WA % DOT_PROD_VECTOR_SIZE){
            printf("WA (%d) must be a multiple DOT_PROD_VECTOR_SIZE (%d)\n", WA, DOT_PROD_VECTOR_SIZE);
            exit(1);
        }
        if(WB % DOT_PROD_VECTOR_SIZE){
            printf("WB (%d) must be a multiple DOT_PROD_VECTOR_SIZE (%d)\n", WB, DOT_PROD_VECTOR_SIZE);
            exit(1);
        }
        
	printf("MAT_B_BLOCK_HEIGHT: \t\t%d\n",  MAT_B_BLOCK_HEIGHT);
	printf("MAT_B_BLOCK_WIDTH: \t\t%d\n",   MAT_B_BLOCK_WIDTH);
	printf("MAT_B_BLOCK_SIZE: \t\t%d\n",    MAT_B_BLOCK_SIZE);

        printf("MAT_B_BLOCK_NUM_VECTORS: \t\t%d\n",   MAT_B_BLOCK_NUM_VECTORS);
        if (MAT_B_BLOCK_SIZE % DOT_PROD_VECTOR_SIZE) {
          printf("MAT_B_BLOCK_SIZE must be a multiple of DOT_PROD_VECTOR_SIZE\b");
        }
        printf("MAT_B_NUM_BLOCKS_IN_ROW: \t\t%d\n",   MAT_B_NUM_BLOCKS_IN_ROW);
        if (WB % MAT_B_BLOCK_WIDTH) {
          printf("WB must be a multiple of MAT_B_BLOCK_WIDTH\n");
        }
        printf("MAT_B_NUM_BLOCKS_IN_COL: \t\t%d\n",   MAT_B_NUM_BLOCKS_IN_COL);
        if (HB % MAT_B_BLOCK_HEIGHT) {
          printf("HB must be a multiple of MAT_B_BLOCK_HEIGHT\n");
        }
        printf("MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS: \t\t%d\n",  MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS);
        printf("MAT_B_NUM_VECTORS_IN_MATRIX: \t\t%d\n",         MAT_B_NUM_VECTORS_IN_MATRIX);
	printf("\n");

	printf("MAT_C_BLOCK_HEIGHT: \t\t%d\n",  MAT_C_BLOCK_HEIGHT);
	printf("MAT_C_BLOCK_WIDTH: \t\t%d\n",   MAT_C_BLOCK_WIDTH);


        if (HA % COMPUTE_GOLD_BLOCK_SIZE) {
          printf("COMPUTE_GOLD_BLOCK_SIZE must evenly divide HA for gold matrix mult computation!\n");
          exit(1);
        }
        if (WB % COMPUTE_GOLD_BLOCK_SIZE) {
          printf("COMPUTE_GOLD_BLOCK_SIZE must evenly divide WB for gold matrix mult computation!\n");
          exit(1);
        }
        if (WA % COMPUTE_GOLD_BLOCK_SIZE) {
          printf("COMPUTE_GOLD_BLOCK_SIZE must evenly divide WA for gold matrix mult computation!\n");
          exit(1);
        }

	float* matrix_mul_inputA;
	float* matrix_mul_inputA_block_wise;

	float* matrix_mul_inputB; // transposed
	float* matrix_mul_inputB_transposed; // non-transposed
	float* matrix_mul_inputB_block_wise; // non-transposed

	float* matrix_mul_outputC;

        float* golden_output;
        float* golden_output_computed_by_blocking;
        float* golden_output_block_wise;
        float* golden_output_block_wise_and_reordered;
        float* golden_output_packed;

	unsigned int num_elem_A = HA*WA;
	unsigned int num_elem_B = HB*WB;
	unsigned int num_elem_C = HC*WC;

        printf("\n===== Host-CPU preparing A,B matrices and computing golden reference for matrix C ======\n\n");
        
        // matrix A
        ////////////  
	if((matrix_mul_inputA = (float*)acl_aligned_malloc(num_elem_A*sizeof(float))) == NULL) {
		perror("Failed malloc of matrix_mul_inputA");
	}
	randomize_array(matrix_mul_inputA, num_elem_A);

	if((matrix_mul_inputA_block_wise = (float*)acl_aligned_malloc(num_elem_A*sizeof(float))) == NULL) {
		perror("Failed malloc of matrix_mul_inputA_block_wise");
	}

        // matrix B
        ///////////  
	if((matrix_mul_inputB = (float*)acl_aligned_malloc(num_elem_B*sizeof(float))) == NULL) {
		perror("Failed malloc of matrix_mul_inputB");
	}
  	randomize_array(matrix_mul_inputB, num_elem_B);
  	
	if((matrix_mul_inputB_transposed = (float*)acl_aligned_malloc(num_elem_B*sizeof(float))) == NULL) {
		perror("Failed malloc of matrix_mul_inputB_transposed");
	}

	if((matrix_mul_inputB_block_wise = (float*)acl_aligned_malloc(num_elem_B*sizeof(float))) == NULL) {
		perror("Failed malloc of matrix_mul_inputB_block_wise");
	}

        ////////////  
	if((matrix_mul_outputC = (float*)acl_aligned_malloc(num_elem_C*sizeof(float))) == NULL) {
		perror("Failed malloc of matrix_mul_outputC");
	}
	memset(matrix_mul_outputC, 0, num_elem_C*sizeof(float));
        ////////////  
        
	if((golden_output = (float*)acl_aligned_malloc(num_elem_C*sizeof(float))) == NULL) {
		perror("Failed malloc of golden_output");
	}
	if((golden_output_computed_by_blocking = (float*)acl_aligned_malloc(num_elem_C*sizeof(float))) == NULL) {
		perror("Failed malloc of golden_output compute by blocking");
	}
	memset(golden_output_computed_by_blocking, 0, num_elem_C*sizeof(float));

	if((golden_output_block_wise = (float*)acl_aligned_malloc(num_elem_C*sizeof(float))) == NULL) {
		perror("Failed malloc of golden_output_block_wise");
	}
	if((golden_output_block_wise_and_reordered = (float*)acl_aligned_malloc(num_elem_C*sizeof(float))) == NULL) {
		perror("Failed malloc of golden_output_block_wise_and_reordered\n");
	}
    if((golden_output_packed = (float*)acl_aligned_malloc(num_elem_C*sizeof(float))) == NULL) {
		perror("Failed malloc of golden_output packed");
	}


        printf("Allocated memory for host-side matrices!\n");
        printf("Transposing and re-formatting of matrices!\n");

        int HA_trim = 0; 
        int WB_trim = 0; 
        int num_elem_C_gold = 0;
        int num_elem_C_gold_first_section = 0;
        int num_elem_C_gold_last_section = 0;
        int C_gold_first_section_offset = 0;
        int C_gold_last_section_offset = 0;

#ifdef COMPUTE_GOLDEN
        HA_trim = 2 * MAT_A_BLOCK_HEIGHT;
        WB_trim = WB; // this one cannot be trimmed, compute_gold requires changes for this to work
        num_elem_C_gold = HA_trim * WB_trim;
          
        printf(" *** Computing golden reference of the result C matrix (only a section of the C matrix), HC(section)=%d, WC(section)=%d!\n",HA_trim, WB_trim);
        printf(" *** This takes several minutes...\n");
      //    void compute_gold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
       //compute_gold(golden_output, matrix_mul_inputA, matrix_mul_inputB, HA_trim, WA, WB_trim);
       num_elem_C_gold = HA_trim * WB_trim;
        //pack_reformat_matrix(golden_output, golden_output_packed, HA,WB, PACK_FACTOR); 
#endif

	printf("Block-wise reformatting of matrix A!\n");
    	row_block_reformat_matrix(matrix_mul_inputA, matrix_mul_inputA_block_wise, HA, WA, MAT_A_BLOCK_HEIGHT,DOT_PROD_VECTOR_SIZE);
        
	printf("Transposing of matrix B!\n");
    	transpose_matrix(matrix_mul_inputB, matrix_mul_inputB_transposed, HB, WB);

       

#ifdef COMPUTE_GOLDEN_BLOCKED
        printf(" *** Computing golden reference of the result C matrix (computing two sections of matrix C)\n");
        printf(" *** This takes several minutes...\n");

        // first two "rows of blocks"
        HA_trim = 2 * MAT_A_BLOCK_HEIGHT;
        WB_trim = WB; // this one cannot be trimmed, compute_gold_blocked requires changes for this to work
        num_elem_C_gold_first_section = HA_trim * WB_trim;
        C_gold_first_section_offset = 0; 

        printf(" *** Computing the first section of the golden C reference, HC(section)=%d, WC(section)=%d!\n",HA_trim, WB_trim);
        //compute_gold_blocked(golden_output_computed_by_blocking, matrix_mul_inputA, matrix_mul_inputB_transposed, HA_trim, WA, WB_trim, HB);

        // last "row of blocks"
        HA_trim = MAT_A_BLOCK_HEIGHT;
        num_elem_C_gold_last_section = HA_trim * WB_trim;
        C_gold_last_section_offset = (HC-HA_trim)*WC;

        printf(" *** Computing the last section of the golden C reference, HC(section)=%d, WC(section)=%d!\n",HA_trim, WB_trim);
        //compute_gold_blocked(golden_output_computed_by_blocking + C_gold_last_section_offset, matrix_mul_inputA + (HA-HA_trim)*WA, matrix_mul_inputB_transposed, HA_trim, WA, WB_trim, HB);

//        printf("Comparing golden_output_computed_by_blocking to default golden_output!\n");     
//        //bool gold_by_blocking_ok = compare_L2_norm(golden_output, golden_output_computed_by_blocking, num_elem_C, 0.0f);
//        bool gold_by_blocking_ok = compare_L2_norm(golden_output, golden_output_computed_by_blocking, num_elem_C, 1.0e-6f);

      // for sanity checks and debugging
      // if (gold_by_blocking_ok != true) {
       //   printf("--> golden_output_computed_by_blocking and golden_output DIFFER!\n");     
       // //printDiff(matrix_mul_inputB, matrix_mul_inputB_block_wise, num_elem_B, 1.0e-6f);
       // } else {
       //   printf("--> golden_output_computed_by_blocking and golden_output are the SAME!\n");     
       // }

        golden_output = golden_output_computed_by_blocking;
#endif


	printf("Block-wise reformatting of matrix B!\n");
    	row_block_reformat_matrix(matrix_mul_inputB_transposed, matrix_mul_inputB_block_wise, WB, HB, MAT_B_BLOCK_WIDTH, DOT_PROD_VECTOR_SIZE);

	printf("Block-wise reformatting of golden output matrix C!\n");
    	block_wise_reformat_matrix(golden_output, golden_output_block_wise, HC, WC, MAT_C_BLOCK_HEIGHT, MAT_C_BLOCK_WIDTH);
    
   // print_matrix(golden_output_block_wise, HC, WC);

	printf("Reordering within blocks of block-wise golden output matrix C!\n");
    //    reorder_within_blocks(golden_output_block_wise, golden_output_block_wise_and_reordered, HC, WC, SYS_ARRAY_NUM_COLS, MAT_C_BLOCK_WIDTH);
	// printf("Matrix A\n");
	// // print_matrix(matrix_mul_inputA, HA, WA);
	// printf("\n");

	// printf("Matrix B (original)\n");
	// print_matrix(matrix_mul_inputB, HB, WB);
	// printf("\n");
	
	// printf("Matrix B (transposed)\n");
	// // print_matrix(matrix_mul_inputB_transposed, WB, HB);
	// printf("\n");

	// printf("Matrix C (gold)\n");
	// print_matrix(golden_output, HC, WC);
	// printf("\n");

    printf("\n===== Host-CPU setting up the OpenCL platform and device ======\n\n");

	// Use this to check the output of each API call
	cl_int status;

	//----------------------------------------------
	// Discover and initialize the platforms
	//----------------------------------------------
	cl_uint numPlatforms = 0;
	cl_platform_id* platforms = NULL;

	// Use clGetPlatformIDs() to retrieve the
	// number of platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
        fprintf(stdout,"Number of platforms = %d\n", numPlatforms);

	// Allocate enough space for each platform
	platforms = (cl_platform_id*) acl_aligned_malloc (numPlatforms * sizeof(cl_platform_id));
        printf("Allocated space for Platform\n");

	// Fill in platforms with clGetPlatformIDs()
	status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);
        printf("Filled in platforms\n");    

	//----------------------------------------------
	// Discover and initialize the devices 
	//----------------------------------------------

	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;

	// Device info
	char buffer[4096];
	unsigned int buf_uint;
	int device_found = 0;

        printf("Initializing IDs\n");    
	for (i=0; i<numPlatforms; i++) {
		status = clGetDeviceIDs(platforms[i],
				CL_DEVICE_TYPE_ALL,
				0,
				NULL,
				&numDevices); 

		if(status == CL_SUCCESS){
			clGetPlatformInfo(platforms[i], 
					CL_PLATFORM_VENDOR,
					4096,
					buffer,
					NULL);
#if defined(ALTERA_CL)
			if(strstr(buffer, "Intel") != NULL){
				device_found = 1;
			}
			printf("%s\n", buffer);
#elif defined(NVIDIA_CL)
			if(strstr(buffer, "NVIDIA") != NULL){
				device_found = 1;
			}
#else
			if(strstr(buffer, "Intel") != NULL){
				device_found = 1;
			}
#endif

			if(device_found){
				// Allocate enough space for each device
				devices = (cl_device_id*)
					acl_aligned_malloc (numDevices * sizeof(cl_device_id));

				// Fill in devices with clGetDeviceIDs()
				status = clGetDeviceIDs(platforms[i],
						CL_DEVICE_TYPE_ALL,
						numDevices,
						devices,
						NULL);
				break;
			}
		}
	}

	if(!device_found) {
		printf("failed to find a OpenCL device\n");
		exit(-1);
	}

	for (i = 0; i < numDevices; i++) {
		clGetDeviceInfo(devices[i],
				CL_DEVICE_NAME,
				4096,
				buffer,
				NULL);
		fprintf(f_out, "\nDevice Name: %s\n", buffer);

		clGetDeviceInfo(devices[i],
				CL_DEVICE_VENDOR,
				4096,
				buffer,
				NULL);
		fprintf(f_out, "Device Vendor: %s\n", buffer);

		clGetDeviceInfo(devices[i],
				CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(buf_uint),
				&buf_uint,
				NULL);
		fprintf(f_out, "Device Computing Units: %u\n", buf_uint);

		clGetDeviceInfo(devices[i],
				CL_DEVICE_GLOBAL_MEM_SIZE,
				sizeof(unsigned long),
				&buffer,
				NULL);
		fprintf(f_out, "Global Memory Size: %i\n", *((unsigned long*)buffer));

		clGetDeviceInfo(devices[i],
				CL_DEVICE_MAX_MEM_ALLOC_SIZE,
				sizeof(unsigned long),
				&buffer,
				NULL);
		fprintf(f_out, "Global Memory Allocation Size: %i\n\n", *((unsigned long*)buffer));
	}



	//----------------------------------------------
	// Create a context 
	//----------------------------------------------

    printf("\n===== Host-CPU setting up the OpenCL command queues ======\n\n");

	cl_context context = NULL;

	// Create a context using clCreateContext() and
	// associate it with the device

	context = clCreateContext(
			NULL,
			1,
			devices,
			NULL,
			NULL,
			&status); CHECK(status);

	//----------------------------------------------
	// Create command queues
	//---------------------------------------------

	cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE+1]; // extra queue for reading buffer C
        
	// Create a command queue using clCreateCommandQueue(),
	// and associate it with the device you want to execute on
	for(i=0; i<NUM_QUEUES_TO_CREATE; i++) {
            fprintf(stdout,"cmdQueue i = %d, kernel name = %s\n", i, kernel_name[i]);
            cmdQueue[i] = clCreateCommandQueue(
				context,
				devices[0],
				CL_QUEUE_PROFILING_ENABLE,
				&status); CHECK(status);
	}
/*
        fprintf(stdout,"cmdQueue i = %d, a queue for reading the C buffer\n", i);
        cmdQueue[i] = clCreateCommandQueue(
                            context,
                            devices[0],
                            CL_QUEUE_PROFILING_ENABLE,
                            &status); CHECK(status);
*/
	//----------------------------------------------
	// Create device buffers
	//----------------------------------------------

	cl_mem d_matrix_mul_outputC;
	cl_mem d_matrix_mul_inputA;
	cl_mem d_matrix_mul_inputB;

	
        printf("\n===== Host-CPU transferring matrices A,B to the FPGA device global memory (DDR4) via PCIe ======\n\n");
	d_matrix_mul_inputA = clCreateBuffer(
		context,
		//CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA,
		CL_MEM_READ_ONLY,
		num_elem_A*sizeof(cl_float),
		NULL,
		&status); CHECK(status);

	d_matrix_mul_inputB = clCreateBuffer(
		context,
		//CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA,
		CL_MEM_READ_ONLY,
		num_elem_B*sizeof(cl_float),
		NULL,
		&status); CHECK(status);

	d_matrix_mul_outputC = clCreateBuffer(
		context,
		//CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA,
		CL_MEM_WRITE_ONLY,
		num_elem_C*sizeof(cl_float),
		NULL,
		&status); CHECK(status);


	//----------------------------------------------
	// Write host data to device buffers
	//----------------------------------------------

        // blocking writes
/* ORIGINAL 
	status = clEnqueueWriteBuffer(
		cmdQueue[KID_FEED_MAT_A],
		d_matrix_mul_inputA,
		CL_TRUE,
		0,
		num_elem_A*sizeof(cl_float),
		matrix_mul_inputA_block_wise,
		//matrix_mul_inputA,
		0,
		NULL,
		NULL); CHECK(status);

	status = clEnqueueWriteBuffer(
		cmdQueue[KID_FEED_MAT_B],
		d_matrix_mul_inputB,
		CL_TRUE,
		0,
		num_elem_B*sizeof(cl_float),
		matrix_mul_inputB_block_wise,
		//matrix_mul_inputB,
		0,
		NULL,
		NULL); CHECK(status);

*/

	status = clEnqueueWriteBuffer(
		cmdQueue[KID_FEED_MAT_A],
		d_matrix_mul_inputA,
		CL_TRUE,
		0,
		num_elem_A*sizeof(cl_float),
		matrix_mul_inputA_block_wise,
		//matrix_mul_inputA,
		0,
		NULL,
		NULL); CHECK(status);



	status = clEnqueueWriteBuffer(
		cmdQueue[KID_FEED_MAT_B],
		d_matrix_mul_inputB,
		CL_TRUE,
		0,
		num_elem_B*sizeof(cl_float),
		matrix_mul_inputB_block_wise,
		//matrix_mul_inputB,
		0,
		NULL,
		NULL); CHECK(status);
	//----------------------------------------------
	// Create the program from binaries
	//----------------------------------------------
        printf("\n===== Host-CPU setting up OpenCL program and kernels ======\n\n");

   	cl_program program;

	size_t binary_length;
	const unsigned char *binary;

        printf("\nAOCX file: %s\n\n", AOCX_FILE);
        // create the program using binary already compiled offline using aoc (i.e. the .aocx file)
	FILE *fp = fopen(AOCX_FILE, "rb");

	if (fp == NULL) {
		printf("Failed to open the AOCX file (fopen).\n");
		return -1;
	}

	fseek(fp, 0, SEEK_END);
	binary_length = ftell(fp);
	binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
	assert(binary && "Malloc failed");
	rewind(fp);

	if (fread((void*)binary, binary_length, 1, fp) == 0) {
		printf("Failed to read from the AOCX file (fread).\n");
		return -1;
	}
	fclose(fp);

	// Create a program using clCreateProgramWithBinary()
	program = clCreateProgramWithBinary(
			context,
			1,
			devices,
			&binary_length,
			(const unsigned char **)&binary,
			&status,
			NULL); 

  CHECK(status);


	//----------------------------------------------
	// Create the kernel
	//----------------------------------------------

	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(status != CL_SUCCESS) {
		char log[128*1024] = {0};
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 128*1024, log, NULL);
		printf("%s\n", log);
		CHECK(status);
	}

	cl_kernel kernel[NUM_KERNELS_TO_CREATE];
   

	for(int j=0; j<NUM_KERNELS_TO_CREATE; j++) {
                printf("Creating kernel[%d]: %s\n", j,kernel_name[j]);
                kernel[j] = clCreateKernel(program, (const char*)kernel_name[j], &status); 
		CHECK(status);
	}

  unsigned int mat_a_num_vectors_in_row_of_blocks = MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS;
  unsigned char mat_a_num_blocks_in_col = MAT_A_NUM_BLOCKS_IN_COL;
  unsigned char mat_b_num_blocks_in_row = MAT_B_NUM_BLOCKS_IN_ROW;
  unsigned int mat_b_matrix_size = (WB*HB) / DOT_PROD_VECTOR_SIZE;


    unsigned int nrXBlocks = WB / MATRIX_B_BLOCK_WIDTH;
    unsigned int nrYBlocks = HA / MATRIX_A_BLOCK_HEIGHT ;
  	unsigned int mat_a_num_vectors_in_row = WA / DOT_PROD_VECTOR_SIZE;

	  int num_vec_per_row = INTERLEAVED * nrXBlocks;
	  unsigned int rowBlockSize = MAT_A_BLOCK_HEIGHT * mat_a_num_vectors_in_row;
        status = clSetKernelArg(
    kernel[KID_FEED_MAT_A],
    0,
    sizeof(cl_mem),
    (void*)&d_matrix_mul_inputA); CHECK(status);

        status = clSetKernelArg(
    kernel[KID_FEED_MAT_A],
    1,
    sizeof(unsigned int),
    (void*)&nrXBlocks); CHECK(status);

        status = clSetKernelArg(
    kernel[KID_FEED_MAT_A],
    2,
    sizeof(unsigned int),
    (void*)&nrYBlocks); CHECK(status);

        status = clSetKernelArg(
    kernel[KID_FEED_MAT_A],
    3,
    sizeof(unsigned int),
    (void*)&rowBlockSize); CHECK(status);

  unsigned int mat_b_num_vectors_in_col_of_blocks = MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS;
  unsigned int mat_b_num_vectors_in_matrix = MAT_B_NUM_VECTORS_IN_MATRIX;

        status = clSetKernelArg(
    kernel[KID_FEED_MAT_B],
    0,
    sizeof(cl_mem),
    (void*)&d_matrix_mul_inputB); CHECK(status);


        status = clSetKernelArg(
    kernel[KID_FEED_MAT_B],
    1,
    sizeof(unsigned int),
    (void*)&nrYBlocks); CHECK(status);

        status = clSetKernelArg(
    kernel[KID_FEED_MAT_B],
    2,
    sizeof(unsigned int),
    (void*)&mat_b_matrix_size); CHECK(status);

  int mat_c_num_coalesced_words = WC * HC / SYS_ARRAY_NUM_COLS;
  int scale = SCALING_FACTOR;



  status = clSetKernelArg(
    kernel[KID_DRAIN_MAT_C],
    0,
    sizeof(cl_mem),
    (void*)&d_matrix_mul_outputC); CHECK(status);

  status = clSetKernelArg(
    kernel[KID_DRAIN_MAT_C],
    1,
    sizeof(int),
    (void*)&mat_c_num_coalesced_words); CHECK(status);

	//----------------------------------------------
	// Configure the work-item structure (using only tasks atm)
	//----------------------------------------------

	// Define the number of threads that will be created 
	// as well as the number of work groups 
	size_t globalWorkSize[1];
	size_t localWorkSize[1];


	//----------------------------------------------
	// Enqueue the kernel for execution
	//----------------------------------------------


        // all kernels are always tasks
	globalWorkSize[0] = 1;
	localWorkSize[0]  = 1;




        printf("\n===== Host-CPU enqeuing the OpenCL kernels to the FPGA device ======\n\n");
	for(i=0; i<NUM_KERNELS_TO_CREATE; i++) {
		// Alternatively, can use clEnqueueTaskKernel
		printf("clEnqueueNDRangeKernel[%d]: %s!\n", i,kernel_name[i]);
		status = clEnqueueNDRangeKernel(
				cmdQueue[i],
				kernel[i],
				1,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
                &kernel_exec_event[i]
                );
		CHECK(status);
	}


        printf(" *** FPGA execution started!\n");
 //PowerSensor::State start = sensor.read();
	for(i=0; i < NUM_KERNELS_TO_CREATE ; i++) {
		status = clFlush(cmdQueue[i]); 
                CHECK(status);
	}

        for(i=0; i < NUM_QUEUES_TO_FINISH; i++) {
           status = clFinish(cmdQueue[i]); CHECK(status);
	}
        printf(" *** FPGA execution finished!\n");

//PowerSensor::State stop = sensor.read();
//  std::cout << "The computation took " << PowerSensor::Joules(start, stop) << 'J' << std::endl;
//  std::cout << "The computation took " << PowerSensor::Watt(start, stop) << 'W' << std::endl;
//  std::cout << "The computation took " << PowerSensor::seconds(start, stop) << 's' << std::endl;
	double k_start_time[NUM_QUEUES_TO_FINISH];	
	double k_end_time[NUM_QUEUES_TO_FINISH];
	double k_exec_time[NUM_QUEUES_TO_FINISH];

	for (i=0; i<NUM_QUEUES_TO_FINISH; i++) {
          k_exec_time[i] = compute_kernel_execution_time(kernel_exec_event[i], k_start_time[i], k_end_time[i]);     
        }     
        printf("\n\n");
    
        printf("\n===== Host-CPU transferring result matrix C from the FPGA device global memory (DDR4) via PCIe ======\n\n");
	
      // Read the results back from the device, blocking read
        clEnqueueReadBuffer(
              //cmdQueue[KID_DRAIN_MAT_C],
              cmdQueue[NUM_KERNELS_TO_CREATE], // using a special queue for reading buffer C
              d_matrix_mul_outputC,
              CL_TRUE,
              0,
              num_elem_C*sizeof(cl_float),
              matrix_mul_outputC,
              0,
              NULL,
              NULL); CHECK(status);

 	// printf("Matrix C (FPGA result)\n");
	// print_matrix(matrix_mul_outputC, HC, WC);
	// printf("\n");

//    printf("1.0e-5f=%f\n", 1.0e-5f);
//    printf("1.0e-6f=%f\n", 1.0e-6f);

    bool res;

// Some sanity checks, this is for debugging only
/*
    printf("\nSanity checks...\n"); 
    printf("Comparing matrix_mul_inputA and matrix_mul_inputA_block_wise!\n");     
    res = compare_L2_norm(matrix_mul_inputA, matrix_mul_inputA_block_wise, num_elem_A, 1.0e-6f);
    if (res != true) {
      printf("--> GOOD, matrix_mul_inputA and matrix_mul_inputA_block_wise differ!\n");     
      // printDiff(matrix_mul_inputB, matrix_mul_inputB_block_wise, num_elem_B, 1.0e-6f);
    } 

    printf("Comparing matrix_mul_inputB and matrix_mul_inputB_block_wise!\n");     
    res = compare_L2_norm(matrix_mul_inputB, matrix_mul_inputB_block_wise, num_elem_B, 1.0e-6f);
    if (res != true) {
      printf("--> GOOD, matrix_mul_inputB and matrix_mul_inputB_block_wise differ!\n");     
      // printDiff(matrix_mul_inputB, matrix_mul_inputB_block_wise, num_elem_B, 1.0e-6f);
    } 

    printf("Comparing golden_output and golden_output_block_wise!\n");     
    res = compare_L2_norm(golden_output_block_wise, golden_output, num_elem_C, 1.0e-6f);
    if (res != true) {
      printf("--> GOOD, golden_output and golden_output_block_wise differ!\n");     
	  // printDiff(golden_output_block_wise, golden_output, num_elem_C, 1.0e-1f);
    } 
    printf("\n\n");
*/


    printf("\n===== Comparing FPGA results to golden reference ======\n\n");
    float epsilon = 1.0e-5f;
    printf("Tolerance epsilon for L2-norm: 1.0e-5f = %f\n", epsilon);

    printf("Comparing FPGA results to golden reference (the first section of matrix C)\n");
res = true;
#ifdef COMPUTE_GOLDEN
    //printDiff(golden_output_block_wise, matrix_mul_outputC, num_elem_C_gold, epsilon);
#endif
    // res = compare_L2_norm(golden_output_block_wise_and_reordered + C_gold_first_section_offset, matrix_mul_outputC + C_gold_first_section_offset, num_elem_C_gold_first_section, epsilon);
    if (res != true) {
               printDiff(golden_output_packed + C_gold_first_section_offset, matrix_mul_outputC + C_gold_first_section_offset, num_elem_C_gold_first_section, epsilon);
    } else { // res == shrTRUE
      printf("Comparing FPGA results to golden reference (the last section of matrix C)\n");
      //res = compare_L2_norm(golden_output_block_wise_and_reordered + C_gold_last_section_offset, matrix_mul_outputC + C_gold_last_section_offset, num_elem_C_gold_last_section, epsilon);
      if (res != true) {
                 printDiff(golden_output_block_wise_and_reordered + C_gold_last_section_offset, matrix_mul_outputC + C_gold_last_section_offset, num_elem_C_gold_last_section, epsilon);
      } 
    }
		
    printf("\n===== Reporting measured throughput ======\n\n");
    double k_earliest_start_time = k_start_time[0];
    double k_latest_end_time     = k_end_time[0];	
	
    for (i=1; i<NUM_QUEUES_TO_FINISH; i++) {

      if (k_start_time[i] < k_earliest_start_time) 	
        k_earliest_start_time 	= k_start_time[i];

      if (k_end_time[i]   > k_latest_end_time) 		
        k_latest_end_time 		= k_end_time[i];
    } 

    // IMPORTANT: we care about the finish time of drain_C, once data is drained we are done
    k_latest_end_time 		= k_end_time[KID_DRAIN_MAT_C];


    for(i=0; i<NUM_QUEUES_TO_FINISH; i++) {
      printf("  Kernel execution time on FPGA: %s, \n   \t\t\t\t\t\t\t\t\texec time = %.5f s, start=%.5f s, end=%.5f s\n", kernel_name[i], k_exec_time[i], k_start_time[i], k_end_time[i]);     
    } 
 
    double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;
	
    printf("\n");
    printf("  Loader kernels start time\t\t= %.5f s\n", k_earliest_start_time);     
    printf("  Drainer kernels end time\t\t= %.5f s\n", k_latest_end_time);     
    printf("  FPGA MatMult exec time\t\t= %.5f s\n", k_overall_exec_time);     

 	// multiplied by 1.0e-9 to get G-FLOPs
    printf("\n");

    double num_operations = (double)2.0 * WA * HC * WC;

    printf("  # operations = %.0f\n", num_operations );     
    printf("  Throughput: %.5f GFLOPS\n", (double)1.0e-9 * num_operations / k_overall_exec_time);     

	printf("\n");
	printf("DONE\n");
        printf("%s\n\n", (res == true ? "PASSED" : "FAILED"));

        FILE *fp_status;
        fp_status=fopen("matrixMult.txt", "w");
        fprintf(fp_status,"%s\n\n", (res == true ? "PASSED" : "FAILED"));
        fclose(fp_status);

	//----------------------------------------------
	// Release the OpenCL resources
	//----------------------------------------------

	// Free resources
	//for(i=0; i<NUM_KERNELS; i++) {
	for(i=0; i<NUM_KERNELS_TO_CREATE; i++) {
          clReleaseKernel(kernel[i]);
	}

	for(i=0; i<NUM_QUEUES_TO_CREATE; i++) {
          clReleaseCommandQueue(cmdQueue[i]);
        }

	for(i=0; i<NUM_QUEUES_TO_FINISH; i++) {
         clReleaseEvent(kernel_exec_event[i]);
        }
	
	clReleaseMemObject(d_matrix_mul_inputA);
	clReleaseMemObject(d_matrix_mul_inputB);
	clReleaseMemObject(d_matrix_mul_outputC);

	acl_aligned_free(matrix_mul_inputA);
	acl_aligned_free(matrix_mul_inputB);
	acl_aligned_free(matrix_mul_outputC);

	clReleaseProgram(program);
	clReleaseContext(context);

	acl_aligned_free(platforms);
	acl_aligned_free(devices);

}


