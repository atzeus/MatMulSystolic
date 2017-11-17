public class MatMulBeauty {

    static final int DOT_PROD_VECTOR_SIZE = MatMul.DOT_PROD_VECTOR_SIZE;
    static final int SYS_ARRAY_NUM_ROWS = MatMul.SYS_ARRAY_NUM_ROWS;
    static final int SYS_ARRAY_NUM_COLS = MatMul.SYS_ARRAY_NUM_COLS;
    static final int INTERLEAVED  = MatMul.INTERLEAVED; // Cols/Rows interleaved
    static final int MATRIX_A_BLOCK_HEIGHT = INTERLEAVED * SYS_ARRAY_NUM_ROWS;
    static final int MATRIX_B_BLOCK_WIDTH = INTERLEAVED * SYS_ARRAY_NUM_COLS;
    static final int INTERLEAVED_SQUARE = INTERLEAVED * INTERLEAVED;

    static final VecFloat VECTOR_ZERO = new VecFloat(0,0,0,0);

    static final int QUEUE_SIZE = 1;


    static <E> WatchFIFO<E>[] channelRow(int size, int queueSize){
        WatchFIFO<E>[] res = new WatchFIFO[size];
        for(int i = 0 ; i < size ; i++){
            res[i] = new WatchFIFO<E>(queueSize);
        }
        return res;
    }

    static <E> WatchFIFO<E>[][] channelGrid(int rows, int cols,int queueSize){
        WatchFIFO<E>[][] res = new WatchFIFO[rows][];
        for(int row = 0 ; row < rows ; row++){
            res[row] = channelRow(cols, queueSize);
        }
        return res;
    }

    static <E> WatchFIFO<E>[][] channelGridSystolic(int queueSize){
        return channelGrid(SYS_ARRAY_NUM_ROWS, SYS_ARRAY_NUM_COLS, queueSize);
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
        ch_data_a = channelGridSystolic(QUEUE_SIZE);
        ch_data_b = channelGridSystolic(QUEUE_SIZE);
        row_feed_chain = channelRow(SYS_ARRAY_NUM_ROWS, QUEUE_SIZE);
        row_feed_to_buf = channelRow(SYS_ARRAY_NUM_ROWS, QUEUE_SIZE);
        col_feed_chain = channelRow(SYS_ARRAY_NUM_COLS, QUEUE_SIZE);
        col_feed_to_buf = channelRow(SYS_ARRAY_NUM_COLS,QUEUE_SIZE);
        ch_data_c = channelGridSystolic(INTERLEAVED_SQUARE);
        ch_drain_c = channelGridSystolic(QUEUE_SIZE);
        col_c_chain = channelRow(SYS_ARRAY_NUM_COLS,QUEUE_SIZE);
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



    static VecFloat[] initVecFloatArray(int size){
        VecFloat[] res = new VecFloat[size];
        for(int i = 0 ; i < size; i++){
            res[i] = VECTOR_ZERO;
        }
        return res;
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
        final int nrXBlocks;
        final int nrYBlocks;
        final int dotProdVecLength;


        LoadMatA(VecFloat[] A, int nrXBlocks, int nrYBlocks, int dotProdVecLength){
            this.A = A;
            this.nrXBlocks = nrXBlocks;
            this.nrYBlocks = nrYBlocks;
            this.dotProdVecLength = dotProdVecLength;
        }

        public void run(){

            syncWithB();
            boolean first = true;
            for(int rowBlock = 0 ; rowBlock < nrYBlocks ; rowBlock++){
                for(int reuse = 0 ; reuse < nrXBlocks ; reuse++){
                    feedRowBlock(rowBlock,first);
                    first = false;
                }
            }
            flushLastBlock();
        }

        void syncWithB(){
            /* Because of buffering, Matrix B feeder start feeding
               actual data after recieving INTERLEAVED vectors per B-buffer element

               However, Matrix A feeder do not need this; they can start after recieving
               zero elements. To sync up them both
               we first feed INTERLEAVED zero vector to each A-buffer element.
               This means a total of SYS_ARRAY_NUM_ROWS * INTERLEAVED zero
               vectors.
            */
            for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS * INTERLEAVED  ; row++){
                VecFloat vec =VECTOR_ZERO;
                boolean new_row_col_pair = false;
                ChannelAData data = new ChannelAData(vec,new_row_col_pair);
                write_channel_intel(row_feed_chain[0],data);
            }
        }


        void feedRowBlock(int rowBlock,boolean first){
            for(int col = 0 ; col < dotProdVecLength; col++){
                for(int row = 0 ; row < MATRIX_A_BLOCK_HEIGHT ; row++){
                    VecFloat vec = A[computeIndex(rowBlock, col, row)];
                    boolean new_row_col_pair = !first && col == 0;
                    ChannelAData data = new ChannelAData(vec,new_row_col_pair);
                    write_channel_intel(row_feed_chain[0],data);
                }
            }
        }

        int computeIndex(int rowBlock, int col, int row_in_block){
            return (rowBlock * MATRIX_A_BLOCK_HEIGHT + row_in_block) * dotProdVecLength + col;
        }

        private void flushLastBlock() {
            /* There is a block of results still in the PEs
               to flush this, we need to first need to indicate the end
               of block with new_row_col_pair is true
               for every result in each PE.
            */
            for(int row = 0 ; row < MATRIX_A_BLOCK_HEIGHT ; row++){
                VecFloat vec =VECTOR_ZERO;
                boolean new_row_col_pair = true;
                ChannelAData data = new ChannelAData(vec,new_row_col_pair);
                write_channel_intel(row_feed_chain[0],data);
            }
            /* to propagate the results through the system, new elements are needed
               to avoid special casing every where, hence we keep feeding elements
               to fully flush the system */
            while(true){
                boolean new_row_col_pair = true;
                ChannelAData data = new ChannelAData(VECTOR_ZERO,new_row_col_pair);
                write_channel_intel(row_feed_chain[0],data);
            }

        }
    }

    // input is transposed
    static class LoadMatB extends Thread {
        final VecFloat[] B;
        final int nrXBlocks;
        final int nrYBlocks;
        final int dotProdVecLength;


        LoadMatB(VecFloat[] B, int nrXBlocks, int nrYBlocks, int dotProdVecLength){
            this.B = B;
            this.nrXBlocks = nrXBlocks;
            this.nrYBlocks = nrYBlocks;
            this.dotProdVecLength = dotProdVecLength;
        }


        public void run(){
            for(int reuse = 0 ; reuse < nrYBlocks; reuse++){
                for(int colBlock = 0 ; colBlock < nrXBlocks ; colBlock++){
                    feedCollumnBlock(colBlock);
                }
            }
            flushLastBlock();
        }


        void feedCollumnBlock(int colBlock){
            for(int row = 0 ; row < dotProdVecLength ; row++){
                for(int col = 0 ; col < MATRIX_B_BLOCK_WIDTH  ; col++){
                    int index = computeIndex(colBlock, row, col);
                    write_channel_intel(col_feed_chain[0],B[index]);
                }
            }
        }

        private int computeIndex(int colBlock, int row, int col) {
            return (colBlock * MATRIX_B_BLOCK_WIDTH + col) *  dotProdVecLength + row;
        }



        private void flushLastBlock() {
                while(true){
//                for(int col = 0 ; col < MATRIX_B_BLOCK_WIDTH ; col++){
                    write_channel_intel(col_feed_chain[0],VECTOR_ZERO);
                }
        }
    }


    // The feeder obtain data from the loader and distribute the data
    // round robin fashion of the buffers

    static class FeedMatA extends Thread {
        final int row;

        FeedMatA(int id){
            this.row = id;
        }

        public void run() {
            final int nrFeedersBelow = (SYS_ARRAY_NUM_ROWS - 1) - row;
            int count = 0;
            while (true) {


                ChannelAData read = read_channel_intel(row_feed_chain[row]);
                write_channel_intel(row_feed_to_buf[row], read);
                for (int feeder = 0; feeder < nrFeedersBelow; feeder++) {
                    read = read_channel_intel(row_feed_chain[row]);
                    write_channel_intel(row_feed_chain[row+1], read);
                }
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
                write_channel_intel(col_feed_to_buf[col], read);
                for (int feeder = 0; feeder < nrFeedersRight; feeder++) {
                    read = read_channel_intel(col_feed_chain[col]);
                    write_channel_intel(col_feed_chain[col+1], read);
                }
            }
        }

    }



    static class Buf_mat_a_kernel extends Thread {

        final int row;

        Buf_mat_a_kernel(int row){
            this.row = row;
        }

        public void run(){
            ChannelAData feed;
            while(true){
                feed = read_channel_intel(row_feed_to_buf[row]);
                for(int reuse = 0 ; reuse < INTERLEAVED ; reuse++){
                    write_channel_intel(ch_data_a[row][0],feed);
                }


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

            float[] interleave_shift = new float[INTERLEAVED_SQUARE];
            for (int i = 0; i < INTERLEAVED_SQUARE; i++) {
                interleave_shift[i] = 0.0f;
            }

            while(true){

                ChannelAData read_A = read_channel_intel(ch_data_a[row][col]);
                if (col < (SYS_ARRAY_NUM_COLS-1)) write_channel_intel(ch_data_a[row][col+1], read_A);

                VecFloat b_data = read_channel_intel(ch_data_b[row][col]);
                if (row < (SYS_ARRAY_NUM_ROWS-1)) write_channel_intel(ch_data_b[row+1][col], b_data);

                float sum;
                if(read_A.new_row_col_pair) {
                   // System.out.printf("Flush! %d %d %d\n", row, col, ch_data_c[row][col].q.remainingCapacity());
                    write_channel_intel(ch_data_c[row][col],interleave_shift[INTERLEAVED_SQUARE -1]);
                    sum = 0f;
                } else {
                    sum = interleave_shift[INTERLEAVED_SQUARE -1];
                }
                for(int d=0; d < DOT_PROD_VECTOR_SIZE; ++d) sum += read_A.data.vals[d] * b_data.vals[d];
                shiftRight(interleave_shift, INTERLEAVED_SQUARE);
                interleave_shift[0] = sum;
            }

        }

        void shiftRight(float[] arr, int length){
            for(int i = length -1 ; i >= 1 ;i--){
                arr[i] = arr[i-1];
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
                // pass on data from above
                for (int i = 0; i < INTERLEAVED * row; i++) {
                    float read = read_channel_intel(ch_drain_c[row - 1][col]);
                    write_channel_intel(ch_drain_c[row][col], read);
                }
                // pass on own data
                for (int i = 0; i < INTERLEAVED; i++) {
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
            int num_vec_per_row = INTERLEAVED * nrXBlocks;
            for(int yblock = 0 ; yblock < nrYBlocks ; yblock++){
                for(int xblock = 0; xblock < nrXBlocks ; xblock++){
                    for(int ylocal = 0 ; ylocal < MATRIX_A_BLOCK_HEIGHT ; ylocal++) {
                        for (int xlocal = 0; xlocal < INTERLEAVED; xlocal++) {
                            int index = ((yblock * MATRIX_A_BLOCK_HEIGHT + ylocal) * num_vec_per_row) + (xblock * INTERLEAVED) + xlocal;
                            VecFloat dataIn = read_channel_intel(col_c_chain[0]);
                            toMem[index] = dataIn;
                        }
                    }
                    System.err.println("Writing yblock : " + yblock + ", xblock : " +xblock);
                }
            }
        }

    }


    public static void run_Mat_mul(VecFloat[] a, int widtha, int heighta, int widthb, VecFloat[] b, VecFloat[] c){
        int nrXBlocks = widthb / MATRIX_B_BLOCK_WIDTH;
        int nrYBlocks = widtha / MATRIX_A_BLOCK_HEIGHT;
        int dotProdVecLength = widtha / DOT_PROD_VECTOR_SIZE;
        new LoadMatA(a, nrXBlocks, nrYBlocks, dotProdVecLength ).start();

        new LoadMatB(b,nrXBlocks, nrYBlocks, dotProdVecLength).start();

        for(int i = 0 ; i < SYS_ARRAY_NUM_ROWS ; i++){
            new FeedMatA(i).start();
            new Buf_mat_a_kernel(i).start();
        }
        for(int i = 0 ; i < SYS_ARRAY_NUM_COLS ; i++){
            new FeedMatB(i).start();
            new Buf_mat_b_kernel(i).start();
        }

        for(int i = 0 ; i < SYS_ARRAY_NUM_ROWS; i++){
            for(int j = 0 ; j < SYS_ARRAY_NUM_COLS; j++) {
                new PE_Kernel(i,j).start();
                new Drain_C(i,j).start();
            }
        }

        for(int i = 0 ; i < SYS_ARRAY_NUM_COLS; i++){
            new Drain_C_cols(i).start();
        }

        new Drain_to_mem(c,nrXBlocks, nrYBlocks).run();

    }

}
