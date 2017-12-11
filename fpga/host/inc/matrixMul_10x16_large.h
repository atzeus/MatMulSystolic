#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// !!! IMPORTANT !!!
// currently changing the matrix dimensions requires a recompile of both the FPGA code and the HOST code

// Matrix dimensions 
#define SCALING_FACTOR 64L

#define HA    (MATRIX_A_BLOCK_HEIGHT * SCALING_FACTOR)       // Matrix A height
#define WA    HA    // Matrix A width

#define HB    WA                                          // Matrix B height
#define WB    (MATRIX_B_BLOCK_WIDTH * SCALING_FACTOR)        // Matrix B width

#define HC HA                                             // Matrix C height
#define WC WB                                             // Matrix C width 

#endif // _MATRIXMUL_H_

