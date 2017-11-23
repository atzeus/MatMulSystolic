#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// !!! IMPORTANT !!!
// currently changing the matrix dimensions requires a recompile of both the FPGA code and the HOST code

// Matrix dimensions 
#define SCALING_FACTOR 4

#define HA    (SYS_ARRAY_NUM_COLS * SYS_ARRAY_NUM_ROWS * INTERLEAVED * SCALING_FACTOR)       // Matrix A height
#define WA    (SYS_ARRAY_NUM_COLS * SYS_ARRAY_NUM_ROWS * INTERLEAVED * SCALING_FACTOR)    // Matrix A width

#define HB    WA                                          // Matrix B height
#define WB    (SYS_ARRAY_NUM_COLS * SYS_ARRAY_NUM_ROWS * INTERLEAVED * SCALING_FACTOR)        // Matrix B width

#define HC HA                                             // Matrix C height
#define WC WB                                             // Matrix C width 

#endif // _MATRIXMUL_H_

