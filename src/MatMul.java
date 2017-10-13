/**
 * Created by atze on 28-9-17.
 */
public class MatMul {


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


    static final int DOT_PROD_VECTOR_SIZE = 3;
    static final int SYS_ARRAY_NUM_ROWS = 3;
    static final int SYS_ARRAY_NUM_COLS = 3;
    static final int INTERLEAVED  = 3; // Cols/Rows interleaved
    static final int MATRIX_A_BLOCK_HEIGHT = INTERLEAVED * SYS_ARRAY_NUM_ROWS;
    static final int MATRIX_B_BLOCK_WIDTH = INTERLEAVED * SYS_ARRAY_NUM_COLS;

    static final int NR_INTERLEAVED = INTERLEAVED * INTERLEAVED;
    static final VecFloat ZERO_ARRAY = new VecFloat(0,0,0,0);
    static final WatchFIFO<ChannelAData>[][] ch_data_a;
    static final WatchFIFO<VecFloat>[][] ch_data_b;
    static final WatchFIFO<Float>[][] ch_data_c;
    static final WatchFIFO<Float>[][] ch_drain_c;

    static final int[][] steps;



    static final WatchFIFO<ChannelAData>[] row_feed_chain;
    static final WatchFIFO<ChannelAData>[] row_feed_to_buf;

    static final int QUEUE_SIZE = 1;

    static final WatchFIFO<VecFloat>[] col_feed_chain;
    static final WatchFIFO<VecFloat>[] col_feed_to_buf;

    static final WatchFIFO<VecFloat>[] ch_data_c_chain;

    static{
        WatchFIFO<ChannelAData>[][] set = new WatchFIFO[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS; col++) {
                set[row][col] = new WatchFIFO<ChannelAData>(QUEUE_SIZE);
            }
        }
        ch_data_a = set;


        WatchFIFO<VecFloat>[][] setb = new WatchFIFO[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS; col++) {
                setb[row][col] = new WatchFIFO<VecFloat>(QUEUE_SIZE);
            }
        }
        ch_data_b = setb;

        WatchFIFO<Float>[][] setc = new WatchFIFO[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS; col++) {
                setc[row][col] = new WatchFIFO<Float>(NR_INTERLEAVED);
            }
        }
        ch_data_c = setc;

        setc = new WatchFIFO[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS; col++) {
                setc[row][col] = new WatchFIFO<Float>(QUEUE_SIZE);
            }
        }
        ch_drain_c = setc;

        WatchFIFO<ChannelAData>[] setf = new WatchFIFO[SYS_ARRAY_NUM_ROWS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            setf[row]= new WatchFIFO<ChannelAData>(QUEUE_SIZE);
        }
        row_feed_chain = setf;

        setf = new WatchFIFO[SYS_ARRAY_NUM_ROWS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            setf[row]= new WatchFIFO<ChannelAData>(QUEUE_SIZE);
        }
        row_feed_to_buf = setf;

        WatchFIFO<VecFloat>[] seth = new WatchFIFO[SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_COLS ; row++ ){
            seth[row]= new WatchFIFO<VecFloat>(QUEUE_SIZE);
        }
        col_feed_chain = seth;

        seth = new WatchFIFO[SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_COLS ; row++ ){
            seth[row]= new WatchFIFO<VecFloat>(QUEUE_SIZE);
        }
        col_feed_to_buf = seth;

        WatchFIFO<VecFloat>[] setx = new WatchFIFO[SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            setx[row]= new WatchFIFO<VecFloat>(QUEUE_SIZE);
        }
        ch_data_c_chain = setx;

        steps = new int[SYS_ARRAY_NUM_ROWS][];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            steps[row]= new int[SYS_ARRAY_NUM_COLS];
            for(int i = 0 ; i < SYS_ARRAY_NUM_COLS ; i++){
                steps[row][i] = 0;
            }
        }
    }


    static <E> void write_channel_intel(WatchFIFO<E> q, E data, String s) throws InterruptedException{
        //System.err.printf("-> write %s\n", s);
        q.put(data);
        //System.err.printf("<- wrote %s\n", s);
    }
    static <E> E read_channel_intele(WatchFIFO<E> q,  String s) throws InterruptedException{
        //System.err.printf("-> wanna read %s\n", s);
        E out = q.take();
        //System.err.printf("<- read %s\n", s);
        return out;
    }

    static <E> E read_channel_intel(WatchFIFO<E> q,  String s) throws InterruptedException{
        //System.err.printf("-> wanna read %s\n", s);
        E out = q.take();
        //System.err.printf("<- read %s\n", s);
        return out;
    }

    static class LoadMatA extends Thread {
        final VecFloat[] A;
        final int mat_a_num_vectors_in_row;
        final int mat_a_num_blocks_in_col;
        final int mat_b_num_blocks_in_row;


        LoadMatA(VecFloat[] A, int mat_a_num_vectors_in_row,int mat_a_num_blocks_in_col, int mat_b_num_blocks_in_row){
            this.A = A;
            this.mat_a_num_vectors_in_row = mat_a_num_vectors_in_row;
            this.mat_a_num_blocks_in_col = mat_a_num_blocks_in_col;
            this.mat_b_num_blocks_in_row = mat_b_num_blocks_in_row;
        }

        public void run() {
            final int mat_a_num_vectors_in_row_of_blocks = mat_a_num_vectors_in_row * MATRIX_A_BLOCK_HEIGHT;
            try {
                boolean first = true;
                for (int yblock = 0; yblock < mat_a_num_blocks_in_col; yblock++) {
                    for (int reuse = 0; reuse < mat_b_num_blocks_in_row; reuse++) {
                            for(int x = 0 ; x < mat_a_num_vectors_in_row ; x++){
                                for(int y = 0 ; y < MATRIX_A_BLOCK_HEIGHT; y++){
                                    int index = (yblock * MATRIX_A_BLOCK_HEIGHT + y) * mat_a_num_vectors_in_row + x ;
                                    ChannelAData write = new ChannelAData(A[index], !first && x == 0);
                                    write_channel_intel(row_feed_chain[0], write, "row_feed_chain[0]");
                                }
                                first = false;
                            }
                            //System.err.printf(" Wrote matrix A BLOCK %d \n", yblock  );
                    }
                }
                // flush last block
                for(int row = 0 ; row <  MATRIX_A_BLOCK_HEIGHT ; row++) {
                    ChannelAData write = new ChannelAData(ZERO_ARRAY, true);
                    write_channel_intel(row_feed_chain[0],write, "row_feed_chain[0]" );
                }
                // Buffer will not flush without new elements
                for(int row = 0 ; row < MATRIX_A_BLOCK_HEIGHT ; row++) {
                    ChannelAData write = new ChannelAData(ZERO_ARRAY, false);
                    write_channel_intel(row_feed_chain[0],write, "row_feed_chain[0]" );
                }
                System.err.println("DONE!!! LOAD MAT A!");

            } catch(InterruptedException e) {
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }
        }
    }


    static class LoadMatB extends Thread {
        final VecFloat[] B;
        final int mat_a_num_vectors_in_row;
        final int mat_a_num_blocks_in_col;
        final int mat_b_num_blocks_in_row;


        LoadMatB(VecFloat[] B, int mat_a_num_vectors_in_row,int mat_a_num_blocks_in_col, int mat_b_num_blocks_in_row){
            this.B = B;
            this.mat_a_num_vectors_in_row = mat_a_num_vectors_in_row;
            this.mat_a_num_blocks_in_col = mat_a_num_blocks_in_col;
            this.mat_b_num_blocks_in_row = mat_b_num_blocks_in_row;
        }

        public void run() {
            final int mat_b_num_vectors_in_col_of_blocks = mat_a_num_vectors_in_row * MATRIX_B_BLOCK_WIDTH;
            final int mat_b_num_vectors_in_col = mat_a_num_vectors_in_row;
            try {
                for (int reuse = 0; reuse < mat_a_num_blocks_in_col; reuse++) {
                    for (int xblock = 0; xblock < mat_b_num_blocks_in_row; xblock++) {
                        for (int y = 0; y < mat_a_num_vectors_in_row; y++) {
                            for (int x = 0; x < MATRIX_B_BLOCK_WIDTH; x++) {
                                int index = (xblock * MATRIX_B_BLOCK_WIDTH + x) * mat_b_num_vectors_in_col + y;
                                //System.err.printf("Loading B  %d %f \n", index, B[index].vals[0]);
                                write_channel_intel(col_feed_chain[0], B[index], "col_feed_chain[0]");
                            }
                        }
                        //System.err.printf(" Wrote matrix A BLOCK %d \n", yblock  );
                    }
                }
                // flush last block
                for(int row = 0 ; row < MATRIX_B_BLOCK_WIDTH *2   ; row++) {
                    write_channel_intel(col_feed_chain[0],ZERO_ARRAY, "col_feed_chain[0]" );
                }
                System.err.println("DONE!!! LOAD MAT B!");
            } catch(InterruptedException e) {
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }
        }
    }

    static class FeedMatA extends Thread {
        final int row;

        FeedMatA(int id){
            this.row = id;
        }

        public void run() {
            try{
                int count = 0;
                int freq = (SYS_ARRAY_NUM_ROWS - 1) - row;
                while(true) {
                    ChannelAData read = read_channel_intel(row_feed_chain[row], "row_feed_chain[" + row + "]");
                    //System.err.printf("Read from feed row chain %d\n", row);
                    if(count == 0) {
                        write_channel_intel(row_feed_to_buf[row], read, "row_feed_to_buf[" + row + "]");
                        count = freq;
                    } else {
                        write_channel_intel(row_feed_chain[row+1], read, "row_feed_to_buf[" + (row + 1) + "]");
                        count--;
                    }
                }

            } catch(InterruptedException e) {
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }
        }
    }


    static class FeedMatB extends Thread {
        final int col;

        FeedMatB(int id){
            this.col = id;
        }

        public void run() {
            try{
                int count = 0;
                int freq = (SYS_ARRAY_NUM_COLS - 1) - col;
                while(true) {
                    VecFloat read = read_channel_intel(col_feed_chain[col], "col_feed_chain[" + col + "]");
                    //System.err.printf("Read from feed col chain %d\n", col);
                    if(count == 0) {
                        write_channel_intel(col_feed_to_buf[col], read, "col_feed_to_buf[" + col + "]");
                        count = freq;
                        //System.err.printf("Wrote to bufc B %f \n", read.vals[0]);
                    } else {
                        write_channel_intel(col_feed_chain[col+1], read, "col_feed_chain[" + (col + 1) + "]");
                        count--;
                    }
                }

            } catch(InterruptedException e) {
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }
        }
    }

    static class Buf_mat_a_kernel extends Thread{

        final int row;
        final ChannelAData[][] buf;

        Buf_mat_a_kernel(int row){
            this.row = row;
            buf = new ChannelAData[2][];
            for(int b = 0 ; b < 2 ; b++){
                buf[b] = new ChannelAData[INTERLEAVED];
                for(int r = 0 ; r < INTERLEAVED ; r++) {
                    buf[b][r] = new ChannelAData(ZERO_ARRAY, false);
                }
            }
        }

        public void run(){
            try{
                // We obtain a new value every COLS_INTERLEAVED steps
                // Each value is also reused COLS_INTERLEAVED steps
                // meaning new values come in at exactly the right rate
                int reuse_cnt = 0;
                int buf_to_read = 0;
                int row_to_read_write = 0;
                while (true) {
                    write_channel_intel(ch_data_a[row][0],buf[buf_to_read][row_to_read_write], "ch_data_a[" + row + "][0]" );
                    //System.err.printf("Wrote new to buf A %f \n",buf[buf_to_read][row_to_read_write].data.vals[0]);
                    if(reuse_cnt == INTERLEAVED - 1) {
                        reuse_cnt = 0;
                        // load once every COLS_INTERLEAVED steps
                        ChannelAData read = read_channel_intel(row_feed_to_buf[row], "row_feed_to_buf[" + row + "]");
                        //System.err.printf("Got new to buf A %d %d\n", buf_to_read, row);
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
            } catch (InterruptedException e) {
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }
        }
    }

    static class Buf_mat_b_kernel extends Thread{

        final int col;
        final VecFloat[][] buf;

        Buf_mat_b_kernel(int col){
            this.col = col;
            buf = new VecFloat[2][];
            for(int b = 0 ; b < 2 ; b++){
                buf[b] = new VecFloat[INTERLEAVED];
                for(int r = 0 ; r < INTERLEAVED ; r++) {
                    buf[b][r] = ZERO_ARRAY;
                }
            }
        }

        public void run(){
            try{
                // We obtain a new value every ROWS_INTERLEAVED steps
                // Each value is also reused ROWS_INTERLEAVED steps
                // meaning new values come in at exactly the right rate (assuming ROWS_INTERLEAVED = COLS_INTERLEAVED)
                int reuse_cnt_col_to_write = 0;
                int buf_to_read = 0;
                int col_to_read = 0;
                while (true) {
                    write_channel_intel(ch_data_b[0][col],buf[buf_to_read][col_to_read],"ch_data_b[0][" + col + "]");
                    //System.err.printf("Wrote new to buf B %d %d %f \n", buf_to_read, col_to_read, buf[buf_to_read][col_to_read].vals[0]);
                    if(col_to_read == INTERLEAVED - 1){
                        col_to_read = 0;
                        // load once every COLS_INTERLEAVED steps
                        VecFloat read = read_channel_intel(col_feed_to_buf[col], "col_feed_to_buf[" + col + "]" );
                        int buf_to_write = 1 - buf_to_read;
                        //System.err.printf("Got new to buf B %d %d %f\n", buf_to_write , reuse_cnt_col_to_write, read.vals[0]);

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
            } catch (InterruptedException e) {
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
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
            try{
                float[] interleave_shift = new float[NR_INTERLEAVED];
                for (int i=0; i < NR_INTERLEAVED  ; i++) {
                    interleave_shift[i] = 0.0f;
                }

                while(true){
                    ChannelAData read_A = read_channel_intel(ch_data_a[row][col], "ch_data_a[" + row + "][" + col + "]");
                    VecFloat a_data = read_A.data;
                    boolean new_col_row_pair = read_A.new_row_col_pair;

                    if (col < (SYS_ARRAY_NUM_COLS-1)) {
                        write_channel_intel(ch_data_a[row][col+1], read_A, "ch_data_a[" + row + "][" + (col  + 1) + "]");
                    }

                    VecFloat b_data = read_channel_intel(ch_data_b[row][col], "ch_data_b[" + row + "][" + col + "]");
                    //System.err.printf("PE Got data %d %f %f\n", steps[row][col], read_A.data.vals[0], b_data.vals[0]);


                    if (row < (SYS_ARRAY_NUM_ROWS-1)) {
                        write_channel_intel(ch_data_b[row+1][col], b_data,"ch_data_b[" + (row  + 1)+ "][" + col   + "]");
                    }

                    if(new_col_row_pair) {
                        System.err.printf("Dumped! %d %d\n", row,col);
                        write_channel_intel(ch_data_c[row][col],interleave_shift[NR_INTERLEAVED-1], "ch_data_c[" + row + "][" + col   + "]");
                    }

                    float sum = 0;


                    for(int d=0; d < DOT_PROD_VECTOR_SIZE; ++d) {
                        sum += a_data.vals[d] * b_data.vals[d];
                    }

                    float accum = sum + (new_col_row_pair ? 0.0f : interleave_shift[NR_INTERLEAVED-1]);

                    for (int i = NR_INTERLEAVED-1; i >= 1; i--) {
                        interleave_shift[i] = interleave_shift[i - 1];
                    }

                    interleave_shift[0] = accum;
                    steps[row][col]++;
                }
            } catch(InterruptedException e){
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
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
            int i = 0;
            int interleaved = 0;
            try {
                while (true) {
                    float read;
                    if (i == 0){
                        read = read_channel_intel(ch_data_c[row][col], "ch_data_c[" + row + "][" + col + "]");
                    } else {
                        read = read_channel_intel(ch_drain_c[row-1][col], "ch_drain_c[" + (row -1 ) + "][" + col + "]");
                    }
                    //if(interleaved == 1 && i == row) {
                     //   System.err.printf("Drain c %d %d %d %d \n", i, interleaved, row, col);
                    //}

                    write_channel_intel(ch_drain_c[row][col], read, "ch_drain_c[" + row  + "][" + col + "]");
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
            } catch (InterruptedException e){
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }

        }
    }

    static class Drain_C_cols extends Thread{
        final int col;

        Drain_C_cols(int id){
            this.col = id;
        }

        public void run (){
            try{
                while(true){
                    //System.err.printf("Before Drain chain  data %d \n",col);
                    float in = read_channel_intel(ch_drain_c[SYS_ARRAY_NUM_ROWS - 1][col], "ch_drain_c[SYS_ARRAY_NUM_ROWS - 1][" + col + "]");
                    //System.err.printf("Drain chain  data %d \n",col);
                    float[] prev_node_data_in = new float[SYS_ARRAY_NUM_COLS];
                    if(col != SYS_ARRAY_NUM_COLS - 1) {
                        prev_node_data_in = read_channel_intel(ch_data_c_chain[col +1], "ch_data_c_chain[" + (col + 1) +"]").vals;
                    }

                    float[] out = new float[SYS_ARRAY_NUM_COLS];

                    for (int i = 0; i < SYS_ARRAY_NUM_COLS - 1; i++) {
                        out[i] = prev_node_data_in[i+1];
                    }

                    out[SYS_ARRAY_NUM_COLS-1] = in;
                    write_channel_intel(ch_data_c_chain[col], new VecFloat(out), "ch_data_c_chain[" + col  +"]");

                }
            } catch (Exception e){
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }
        }
    }

    static class Drain_to_mem extends Thread{

        VecFloat[] toMem;
        final int num_vecs_to_write;

        Drain_to_mem(VecFloat[] toMem, int num_vecs_to_write){
            this.toMem = toMem;
            this.num_vecs_to_write = num_vecs_to_write;
        }

        public void run(){
            try {
                for (int i = 0; i < num_vecs_to_write; i++) {
                    VecFloat dataIn = read_channel_intel(ch_data_c_chain[0], "ch_data_c_chain[0]");
                    System.err.printf("Write data  %d of %d : %f \n",i, num_vecs_to_write, dataIn.vals[0]);
                    toMem[i] = dataIn;
                }
            }  catch (InterruptedException e) {
                System.err.println("Interrupted!! " + e.getMessage());
                System.exit(0);
            }
        }

    }

    public static void run_Mat_mul(VecFloat[] a, int widtha, int heighta, int widthb, VecFloat[] b, VecFloat[] c){
        new LoadMatA(a, widtha/ DOT_PROD_VECTOR_SIZE, heighta / MATRIX_A_BLOCK_HEIGHT,widthb / MATRIX_B_BLOCK_WIDTH ).start();;
        new LoadMatB(b,widtha/ DOT_PROD_VECTOR_SIZE, heighta / MATRIX_A_BLOCK_HEIGHT,widthb / MATRIX_B_BLOCK_WIDTH).start();

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

        new Drain_to_mem(c,c.length).run();

    }

    public static float[][] matrixMulSystolic(float[][] a, float[][] b, int widtha, int heighta, int widthb){
        float[] a_blocked = flatten(a,widtha,heighta);
        VecFloat[] a_blocked_vec = vectorize(a_blocked, DOT_PROD_VECTOR_SIZE);
        b = transpose(b,widthb, widtha);
        float[] b_blocked = flatten(b,widthb,widtha);
        VecFloat[] b_blocked_vec = vectorize(b_blocked, DOT_PROD_VECTOR_SIZE);
        VecFloat[] res = new VecFloat[(heighta * widthb) / SYS_ARRAY_NUM_COLS ];
        run_Mat_mul(a_blocked_vec, widtha , heighta, widthb, b_blocked_vec, res);
        float[] resv = devectorize(res, SYS_ARRAY_NUM_COLS);
        float[][] resw =  deflatten(resv, widthb, heighta);
        return resw;


    }

    public static float[][] matrixMul(float[][] a, float[][] b, int widtha, int heighta, int widthb){
        float[][] res = new float[heighta][];
        for(int row = 0 ; row < heighta; row++){
            res[row] = new float[widthb];
            for(int col = 0 ; col < widthb; col++){
                float sum = 0.0f;
                for(int k = 0 ; k < widtha ; k++){
                    sum+= a[row][k] * b[k][col];
                }
                res[row][col] = sum;
            }
        }
        return res;
    }

    public static void checkem(float[][] a, float[][] b, int widtha, int heighta, int widthb, float err){
        float[][] resNorm= matrixMul(a,b,widtha,heighta,widthb);
        float[][] resSystolic = matrixMulSystolic(a,b,widtha,heighta,widthb);
        int nza = 0;
        int nzb = 0;
        for(int row = 0; row < heighta ; row++){
            for(int col = 0; col < widthb ; col++){
                float dist = Math.abs(resNorm[row][col] - resSystolic[row][col]);
                if(resNorm[row][col] > err){
                    nza++;
                }
                if(resSystolic[row][col] > err){
                    nzb++;
                }
                //if(dist > err){
                    System.out.println((dist > err ? "ERR" : "ok ") + " of " + dist +  "at row " + row + ", col " + col +  " orig: " + resNorm[row][col] + " systolic" + resSystolic[row][col]);
                //}
            }
        }
        System.out.println("Nonzeros : " + nza + " , " + nzb);
    }

    public static float[][] ident(int size) {
        float[][] res = new float[size][];
        for(int row = 0 ; row < size ; row++ ){
            res[row] = new float[size];
            for(int col = 0 ; col < size ; col++){
                res[row][col] = row == col ? 1.0f : 0.0f;
            }
        }
        return res;
    }


    public static float[][] transpose(float[][] in, int width, int height){
        float[][] res = new float[width][];
        for(int row = 0 ; row < width; row++) {
            res[row] = new float[height];
            for (int col = 0; col < height; col++) {
                res[row][col] = in[col][row];
            }
        }
        return res;
    }

    public static float[] flatten(float[][] in, int width , int height){
        float[] res = new float[width * height];
        int i = 0 ;
        for(int y = 0 ; y < height ; y++){
            for(int x = 0 ; x < width ; x++){
                res[i] = in[y][x];
                i++;
            }
        }
        return res;
    }

    public static float[][] deflatten(float in[], int width, int height){
        float[][] res = new float[height][];
        for(int y = 0 ; y < height ; y++){
            res[y] = new float[width];
            for(int x = 0 ; x < width ; x++){
                res[y][x] = in[y * width + x];
            }
        }
        return res;
    }

    public static float[] flattenNBlock(float[][] in, int width, int height, int blockwidth, int blockheight){
        float[] res = new float[width * height];
        int i = 0;
        for(int xblock = 0 ; xblock < width / blockwidth ; xblock++){
            for(int yblock = 0; yblock < height / blockheight; yblock++){
                for(int x = 0 ; x < blockwidth ; x++){
                    for(int y = 0 ; y < blockheight ; y++){
                        res[i] =  in[yblock * blockheight + y][xblock * blockwidth + x];
                    }
                }
            }
        }
        return res;
    }

    public static float[][] deflattenNDeBlock(float[] in, int width, int height, int blockwidth, int blockheight) {
        float[][] res = new float[height][];
        int blocks_in_row = width / blockwidth;
        for(int row = 0 ; row < height ; row++){
            res[row] = new float[width];
            for(int col = 0 ; col < width ; col++){
                int blockindex = (row / blockheight) * blocks_in_row + (col / blockwidth);
                int index_in_block = (row % blockheight) * blockwidth + (col % blockwidth);
                res[row][col] = in[blockindex * (blockwidth * blockheight) + index_in_block];
            }
        }
        return res;
    }

    public static VecFloat[] vectorize(float[] in, int vecsize){
        VecFloat[] res = new VecFloat[in.length / vecsize];
        for(int large = 0 ; large <in.length / vecsize; large++){
            float[] vec = new float[vecsize];
            for(int i = 0 ; i < vecsize; i++){
                vec[i] = in[large * vecsize + i];
            }
            res[large] = new VecFloat(vec);
        }
        return res;
    }

    public static float[] devectorize(VecFloat[] in, int vecsize){
        float[] res = new float[in.length * vecsize];
        for(int large = 0 ; large < in.length ; large++){
            for(int i = 0 ; i < vecsize; i++){
                res[large * vecsize + i] = in[large].vals[i];
            }
        }
        return res;
    }


    static class WatchEm  extends Thread{

        <E>  String status(WatchFIFO<E> q){
            char stat;
            if(!q.taking  && !q.offering){
                stat= '.';
            } else if(q.taking && !q.offering){
                stat= 'T';
            } else if(!q.taking && q.offering){
                stat= 'O';
            } else {
                stat= '?';
            }
            return stat + "";
        }

        <E> String row(WatchFIFO<E>[] qs){
            String res = "";
            for(WatchFIFO<E> q : qs){
                res += status(q);
            }
            return res;
        }

        <E> String grid(WatchFIFO<E>[][] qs){
            String res = "";
            for(WatchFIFO<E>[] q : qs){
                res += row(q) + "\n";
            }
            return res;
        }

        String gridInt(int[][] qs){
            String res = "";
            for(int[] q : qs){
                for(int i : q){
                    res += i + " ";
                }
                res += "\n";
            }
            return res;
        }

        String transpose(String ... els){
            String res = "";
            for(int i = 0 ; i < els[0].length(); i++){
                for(int j = 0 ; j < els.length; j++){
                    res += els[j].charAt(i);
                }
                res += "\n";
            }
            return res;
        }

        String mergeHor(String a, String b){
            String linesa[] = a.split("\\r?\\n");
            String linesb[] = b.split("\\r?\\n");
            String res = "";
            for(int i = 0 ; i < linesa.length ; i++){
                res+= linesa[i] + " " + linesb[i] + "\n";
            }
            return res;
        }

        void visualize(){
            System.err.println("A ");
            System.err.print(mergeHor(transpose(row(row_feed_chain), row(row_feed_chain)), grid(ch_data_a)));
            /*System.err.println(row(row_feed_chain));
            System.err.println(row(row_feed_chain));
            System.err.println();
            System.err.println(grid(ch_data_a));
            */
            System.err.println();
            System.err.println("B");
            System.err.println(row(col_feed_chain));
            //System.err.println(row(col_feed_to_buf));
            System.err.println();
            System.err.println(grid(ch_data_b));

            System.err.println("\n");
            System.err.println("C");
            System.err.println(grid(ch_data_c));
            System.err.println("C drain");
            System.err.println(grid(ch_drain_c));
            System.err.println();
            System.err.println(row(ch_data_c_chain));

            System.err.print(gridInt(steps));

            System.err.println();
        }

        public void run() {
        try {
            while (true) {
                sleep(5000);
                visualize();
            }
        } catch(InterruptedException e){

        }
        }
    }

    public static void main(String[] argv){
        int width = MATRIX_A_BLOCK_HEIGHT * 1 ;
        new WatchEm().start();;
        float[][] ident = ident(width);
        checkem(ident,ident,width,width,width, 0.001f);
        System.out.println("DONE");
        try {
            Thread.sleep(100000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.exit(0);
    }

}
