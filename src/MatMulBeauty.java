public class MatMulBeauty {

    static final int DOT_PROD_VECTOR_SIZE = 1;
    static final int SYS_ARRAY_NUM_ROWS = 4;
    static final int SYS_ARRAY_NUM_COLS = 4;
    static final int INTERLEAVED  = 4; // Cols/Rows interleaved
    static final int MATRIX_A_BLOCK_HEIGHT = INTERLEAVED * SYS_ARRAY_NUM_ROWS;
    static final int MATRIX_B_BLOCK_WIDTH = INTERLEAVED * SYS_ARRAY_NUM_COLS;
    static final int NR_INTERLEAVED = INTERLEAVED * INTERLEAVED;

    static final VecFloat VECTOR_ZERO = new VecFloat(0,0,0,0);

    static final int QUEUE_SIZE = 1;


    static <E> WatchFIFO<E>[] channelRow(int size){
        WatchFIFO<E>[] res = new WatchFIFO[size];
        for(int i = 0 ; i < size ; i++){
            res[i] = new WatchFIFO<E>(QUEUE_SIZE);
        }
        return res;
    }

    static <E> WatchFIFO<E>[][] channelGrid(int rows, int cols){
        WatchFIFO<E>[][] res = new WatchFIFO[rows][];
        for(int row = 0 ; row < rows ; row++){
            res[row] = channelRow(cols);
        }
        return res;
    }

    static <E> WatchFIFO<E>[][] channelGridSystolic(){
        return channelGrid(SYS_ARRAY_NUM_ROWS, SYS_ARRAY_NUM_ROWS);
    }

    static final WatchFIFO<ChannelAData>[] row_feed_chain;
    static final WatchFIFO<ChannelAData>[] row_feed_to_buf;
    static final WatchFIFO<VecFloat>[] col_feed_chain;
    static final WatchFIFO<VecFloat>[] col_feed_to_buf;
    static final WatchFIFO<ChannelAData>[][] ch_data_a;
    static final WatchFIFO<VecFloat>[][] ch_data_b;
    static final WatchFIFO<Float>[][] ch_data_c;
    static final WatchFIFO<Float>[][] ch_drain_c;
    static final WatchFIFO<VecFloat>[] col_c_chain;

    static {
        ch_data_a = channelGridSystolic();
        ch_data_b = channelGridSystolic();
        row_feed_chain = channelRow(SYS_ARRAY_NUM_ROWS);
        row_feed_to_buf = channelRow(SYS_ARRAY_NUM_ROWS);
        col_feed_chain = channelRow(SYS_ARRAY_NUM_COLS);
        col_feed_to_buf = channelRow(SYS_ARRAY_NUM_COLS);
        ch_data_c = channelGridSystolic();
        ch_drain_c = channelGridSystolic();
        col_c_chain = channelRow(SYS_ARRAY_NUM_ROWS);
    }


    static <E> void write_channel_intel(WatchFIFO<E> q, E data){
        try {
            q.put(data);
        } catch (InterruptedException e){
            throw new Error(e.toString());
        }
    }

    static <E> E read_channel_intel(WatchFIFO<E> q){
        try {
            return q.take();
        } catch (InterruptedException e){
            throw new Error(e.toString());
        }
    }


    static class VecFloat {
        final float[] vals;

        VecFloat(float ... vals){
            this.vals = new float[vals.length];
            for(int i = 0 ; i < vals.length ; i++){
                this.vals[i] = vals[i];
            }
        }
    }


    static class ChannelAData {
        final VecFloat data;
        final boolean new_row_col_pair;

        ChannelAData(VecFloat data, boolean new_row_col_pair){
            this.data = data;
            this.new_row_col_pair = new_row_col_pair;
        }
    }


    static class LoadMatA extends Thread {
        final VecFloat[] A;
        final int mat_a_num_vectors_in_row;
        final int mat_a_num_blocks_in_col;
        final int mat_b_num_blocks_in_row;


        LoadMatA(VecFloat[] A, int mat_a_num_vectors_in_row, int mat_a_num_blocks_in_col, int mat_b_num_blocks_in_row){
            this.A = A;
            this.mat_a_num_vectors_in_row = mat_a_num_vectors_in_row;
            this.mat_a_num_blocks_in_col = mat_a_num_blocks_in_col;
            this.mat_b_num_blocks_in_row = mat_b_num_blocks_in_row;
        }

        public void run(){
            for(int rowBlock = 0 ; rowBlock < mat_a_num_blocks_in_col ; rowBlock++){
                for(int reuse = 0 ; reuse < mat_b_num_blocks_in_row ; reuse++){
                    feedRowBlock(rowBlock);
                }
            }
            flushLastBlock();
        }

        private void flushLastBlock() {

        }

        void feedRowBlock(int rowBlock){
            for(int col = 0 ; col < mat_a_num_vectors_in_row; col++){
                for(int row = 0 ; row < MATRIX_A_BLOCK_HEIGHT ; row++){
                    VecFloat vec = A[computeIndex(rowBlock, col, row)];
                    boolean new_row_col_pair = row == 0;
                    ChannelAData data = new ChannelAData(vec,new_row_col_pair);
                    write_channel_intel(row_feed_chain[0],data);
                }
            }
        }

        int computeIndex(int rowBlock, int col, int row_in_block){
            return (rowBlock * MATRIX_A_BLOCK_HEIGHT + row_in_block) + col;
        }
    }


    static class FeedMatA extends Thread {
        final int row;

        FeedMatA(int id){
            this.row = id;
        }

        public void run() {
            final int nrFeedersBelow = (SYS_ARRAY_NUM_ROWS - 1) - row;

            while (true) {
                ChannelAData read = read_channel_intel(row_feed_chain[row]);
                for (int feeder = 0; feeder < nrFeedersBelow; feeder++) {
                    write_channel_intel(row_feed_chain[row], read);
                }
                write_channel_intel(row_feed_to_buf[row], read);
            }
        }

    }




    static class FeedMatB extends Thread {
        final int col;

        FeedMatB(int id){
            this.col = id;
        }

        public void run() {
            final int nrFeedersRight = (SYS_ARRAY_NUM_COLS - 1) - col;

            while (true) {
                VecFloat read = read_channel_intel(col_feed_chain[col]);
                for (int feeder = 0; feeder < nrFeedersRight; feeder++) {
                    write_channel_intel(col_feed_chain[col], read);
                }
                write_channel_intel(col_feed_chain[col], read);
            }
        }

    }

    static VecFloat[] initVecFloatArray(int size){
        VecFloat[] res = new VecFloat[size];
        for(int i = 0 ; i < size; i++){
            res[i] = VECTOR_ZERO;
        }
        return res;
    }

    static class Buf_mat_a_kernel extends Thread {

        final int row;

        Buf_mat_a_kernel(int row){
            this.row = row;
        }

        public void run(){
            ChannelAData feed = new ChannelAData(VECTOR_ZERO, false);
            while(true){
                for(int reuse = 0 ; reuse < INTERLEAVED ; reuse++){
                    write_channel_intel(ch_data_a[row][0],feed);
                }
                feed = read_channel_intel(row_feed_to_buf[row]);
            }
        }
    }


    static class Buf_mat_b_kernel extends Thread {

        final int col;

        Buf_mat_b_kernel(int col){
            this.col = col;
        }


        public void run(){
            VecFloat[][] buf = new VecFloat[2][];
            buf[0] = initVecFloatArray(INTERLEAVED);
            buf[1] = initVecFloatArray(INTERLEAVED);
            int bufIndex = 0;
            while(true){
                for(int reuse = 0 ; reuse < INTERLEAVED ; reuse++){
                    feedInterleaved(buf[bufIndex]);
                    int backBuf = 1 - bufIndex;
                    buf[backBuf][reuse] =read_channel_intel(col_feed_to_buf[col]);
                }
                bufIndex = 1 - bufIndex;
            }
        }

        void feedInterleaved(VecFloat[] buf) {
            for(int i = 0 ; i < INTERLEAVED ; i++){
                write_channel_intel(ch_data_b[0][col],buf[i]);
            }
        }
    }


    static class PE_Kernel extends Thread {
        final int row;
        final int col;

        PE_Kernel(int row, int cols){
            this.row = row;
            this.col = cols;
        }

        public void run() {

            float[] interleave_shift = new float[NR_INTERLEAVED];
            for (int i=0; i < NR_INTERLEAVED  ; i++) {
                interleave_shift[i] = 0.0f;
            }

            while(true){

                ChannelAData read_A = read_channel_intel(ch_data_a[row][col]);
                if (col < (SYS_ARRAY_NUM_COLS-1)) write_channel_intel(ch_data_a[row][col+1], read_A);

                VecFloat b_data = read_channel_intel(ch_data_b[row][col]);
                if (row < (SYS_ARRAY_NUM_ROWS-1)) write_channel_intel(ch_data_b[row+1][col], b_data);

                float sum;
                if(read_A.new_row_col_pair) {
                    write_channel_intel(ch_data_c[row][col],interleave_shift[NR_INTERLEAVED-1]);
                    sum = 0f;
                } else {
                    sum = interleave_shift[NR_INTERLEAVED-1];
                }

                for(int d=0; d < DOT_PROD_VECTOR_SIZE; ++d) sum += read_A.data.vals[d] * b_data.vals[d];
                for (int i = NR_INTERLEAVED-1; i >= 1; i--) interleave_shift[i] = interleave_shift[i - 1];
                interleave_shift[0] = sum;
            }

        }
    }


    static class Drain_C extends Thread{
        final int row, col;

        Drain_C(int row, int col){
            this.row = row;
            this.col = col;
        }

        public void run() {
            while (true) {
                for(int r = 0 ; r < row ; r++){
                    for(int i = 0 ; i < NR_INTERLEAVED ; i++){
                        float read = read_channel_intel(ch_drain_c[row-1][col]);
                        write_channel_intel(ch_drain_c[row][col], read);
                    }
                }
                for(int i = 0 ; i < NR_INTERLEAVED ; i++){
                    float read = read_channel_intel(ch_data_c[row][col]);
                    write_channel_intel(ch_drain_c[row][col], read);
                }
            }

        }
    }


    static class Drain_C_cols extends Thread{
        final int col;

        Drain_C_cols(int id){
            this.col = id;
        }

        public void run (){
            while(true){
                float in = read_channel_intel(ch_drain_c[SYS_ARRAY_NUM_ROWS-1][col]);
                float[] prev_node_data_in = new float[SYS_ARRAY_NUM_COLS];
                if(col != SYS_ARRAY_NUM_COLS - 1)
                    prev_node_data_in = read_channel_intel(col_c_chain[col + 1]).vals;

                float[] out = new float[SYS_ARRAY_NUM_COLS];
                for (int i = col + 1; i < SYS_ARRAY_NUM_COLS; i++)
                    out[i] = prev_node_data_in[i];

                out[col] = in;
                write_channel_intel(col_c_chain[col], new VecFloat(out));


            }

        }
    }


    static class Drain_to_mem extends Thread{

        VecFloat[] toMem;
        final int nrXBlocks;
        final int nrYBlocks;

        Drain_to_mem(VecFloat[] toMem, int nrXBlocks, int nrYBlocks){
            this.toMem = toMem;
            this.nrXBlocks = nrXBlocks;
            this.nrYBlocks = nrYBlocks;
        }

        public void run(){
            int num_vec_per_row = MATRIX_B_BLOCK_WIDTH * nrXBlocks;
            for(int yblock = 0 ; yblock < nrYBlocks ; yblock++){
                for(int xblock = 0; xblock < nrXBlocks ; xblock++){
                    for(int ylocal = 0 ; ylocal < MATRIX_A_BLOCK_HEIGHT ; ylocal++) {
                        for (int xlocal = 0; xlocal < MATRIX_B_BLOCK_WIDTH; xlocal++) {
                            int index = ((yblock * MATRIX_A_BLOCK_HEIGHT + ylocal) * num_vec_per_row + (xblock * MATRIX_B_BLOCK_WIDTH) + xlocal;
                            VecFloat dataIn = read_channel_intel(col_c_chain[0]);
                            toMem[index] = dataIn;
                        }
                    }
                }
            }
        }

    }

}
