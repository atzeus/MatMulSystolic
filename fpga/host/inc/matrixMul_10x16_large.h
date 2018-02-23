#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// !!! IMPORTANT !!!
// currently changing the matrix dimensions requires a recompile of both the FPGA code and the HOST code

// Matrix dimensions 
#define SCALING_FACTOR 2L

#define HA    (MATRIX_A_BLOCK_HEIGHT * SCALING_FACTOR)       // Matrix A height
#define WA    HA    // Matrix A width

#define HB    WA                                          // Matrix B height
#define WB    (MATRIX_B_BLOCK_WIDTH * SCALING_FACTOR)        // Matrix B width

#define HC HA                                             // Matrix C height
#define WC WB                                             // Matrix C width 


#define MAT_A_NUM_BLOCKS_IN_ROW             (WA / MAT_A_BLOCK_WIDTH)
#define MAT_A_NUM_BLOCKS_IN_COL             (HA / MAT_A_BLOCK_HEIGHT)
#define MAT_A_NUM_VECTORS_IN_ROW_OF_BLOCKS  (MAT_A_NUM_BLOCKS_IN_ROW * MAT_A_BLOCK_NUM_VECTORS)

#define MAT_B_NUM_BLOCKS_IN_ROW             (WB / MAT_B_BLOCK_WIDTH)
#define MAT_B_NUM_BLOCKS_IN_COL             (HB / MAT_B_BLOCK_HEIGHT)
#define MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS  (MAT_B_NUM_BLOCKS_IN_COL * MAT_B_BLOCK_NUM_VECTORS)
#define MAT_B_NUM_VECTORS_IN_MATRIX         (MAT_B_NUM_VECTORS_IN_COL_OF_BLOCKS * MAT_B_NUM_BLOCKS_IN_ROW)


#endif // _MATRIXMUL_H_

