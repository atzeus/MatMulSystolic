import com.sun.org.apache.bcel.internal.generic.FLOAD;

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


    static final int DOT_PROD_VECTOR_SIZE = 1;
    static final int SYS_ARRAY_NUM_ROWS = 4;
    static final int SYS_ARRAY_NUM_COLS = 4;
    static final int INTERLEAVED  = 4; // Cols/Rows interleaved
    static final int MATRIX_A_BLOCK_HEIGHT = INTERLEAVED * SYS_ARRAY_NUM_ROWS;
    static final int MATRIX_B_BLOCK_WIDTH = INTERLEAVED * SYS_ARRAY_NUM_COLS;

    static final int NR_INTERLEAVED = INTERLEAVED * INTERLEAVED;
    static final VecFloat VECTOR_ZERO = new VecFloat(0,0,0,0);
    static final WatchFIFO<ChannelAData>[][] ch_data_a;
    static final WatchFIFO<ChannelAData>[] ch_data_a_border;
    static final WatchFIFO<VecFloat>[][] ch_data_b;
    static final WatchFIFO<VecFloat>[] ch_data_b_border;
    static final WatchFIFO<Float>[][] ch_data_c;
    static final WatchFIFO<Float>[][] ch_drain_c;

    static final WatchFIFO<Float>[] ch_drain_c_border;

    static final int[][] steps;

    static final WatchFIFO<ChannelAData> row_feed_chain_border;

    static final WatchFIFO<ChannelAData>[] row_feed_chain;
    static final WatchFIFO<ChannelAData>[] row_feed_to_buf;

    static final int QUEUE_SIZE = 1;

    static final WatchFIFO<VecFloat> col_feed_chain_border;

    static final WatchFIFO<VecFloat>[] col_feed_chain;
    static final WatchFIFO<VecFloat>[] col_feed_to_buf;

    static final WatchFIFO<VecFloat>[] col_c_chain;
    static final WatchFIFO<VecFloat> col_c_chain_border;

    static{
        WatchFIFO<ChannelAData>[][] set = new WatchFIFO[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS-1];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS - 1; col++) {
                set[row][col] = new WatchFIFO<ChannelAData>(QUEUE_SIZE);
            }
        }
        ch_data_a = set;

        WatchFIFO<ChannelAData>[] seta = new WatchFIFO[SYS_ARRAY_NUM_ROWS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            seta[row] = new WatchFIFO<ChannelAData>(QUEUE_SIZE);
        }
        ch_data_a_border = seta;


        WatchFIFO<VecFloat>[][] setb = new WatchFIFO[SYS_ARRAY_NUM_ROWS-1][SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS - 1 ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS; col++) {
                setb[row][col] = new WatchFIFO<VecFloat>(QUEUE_SIZE);
            }
        }
        ch_data_b = setb;

        WatchFIFO<VecFloat>[] setax = new WatchFIFO[SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_COLS ; row++ ){
            setax[row] = new WatchFIFO<VecFloat>(QUEUE_SIZE);
        }
        ch_data_b_border = setax;


        WatchFIFO<Float>[][] setc = new WatchFIFO[SYS_ARRAY_NUM_ROWS][SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS; col++) {
                setc[row][col] = new WatchFIFO<Float>(NR_INTERLEAVED);
            }
        }
        ch_data_c = setc;

        WatchFIFO<Float>[] setaf = new WatchFIFO[SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_COLS ; row++ ){
            setaf[row] = new WatchFIFO<Float>(QUEUE_SIZE);
        }
        ch_drain_c_border = setaf;

        setc = new WatchFIFO[SYS_ARRAY_NUM_ROWS-1][SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS -1 ; row++ ){
            for(int col = 0 ; col < SYS_ARRAY_NUM_COLS; col++) {
                setc[row][col] = new WatchFIFO<Float>(QUEUE_SIZE);
            }
        }
        ch_drain_c = setc;

        WatchFIFO<ChannelAData>[] setf = new WatchFIFO[SYS_ARRAY_NUM_ROWS-1];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS  -1; row++ ){
            setf[row]= new WatchFIFO<ChannelAData>(QUEUE_SIZE);
        }
        row_feed_chain = setf;

        WatchFIFO<ChannelAData> n = new WatchFIFO<>(QUEUE_SIZE);
        row_feed_chain_border = n;

        setf = new WatchFIFO[SYS_ARRAY_NUM_ROWS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            setf[row]= new WatchFIFO<ChannelAData>(QUEUE_SIZE);
        }
        row_feed_to_buf = setf;

        WatchFIFO<VecFloat>[] seth = new WatchFIFO[SYS_ARRAY_NUM_COLS-1];
        for(int row = 0 ; row < SYS_ARRAY_NUM_COLS - 1 ; row++ ){
            seth[row]= new WatchFIFO<VecFloat>(QUEUE_SIZE);
        }
        col_feed_chain = seth;


        WatchFIFO<VecFloat> nx = new WatchFIFO<>(QUEUE_SIZE);
        col_feed_chain_border = nx;

        seth = new WatchFIFO[SYS_ARRAY_NUM_COLS];
        for(int row = 0 ; row < SYS_ARRAY_NUM_COLS ; row++ ){
            seth[row]= new WatchFIFO<VecFloat>(QUEUE_SIZE);
        }
        col_feed_to_buf = seth;

        WatchFIFO<VecFloat>[] setx = new WatchFIFO[SYS_ARRAY_NUM_COLS-1];
        for(int row = 0 ; row < SYS_ARRAY_NUM_COLS -1 ; row++ ){
            setx[row]= new WatchFIFO<VecFloat>(QUEUE_SIZE);
        }
        col_c_chain = setx;

        col_c_chain_border = new WatchFIFO<>(QUEUE_SIZE);

        steps = new int[SYS_ARRAY_NUM_ROWS][];
        for(int row = 0 ; row < SYS_ARRAY_NUM_ROWS ; row++ ){
            steps[row]= new int[SYS_ARRAY_NUM_COLS];
            for(int i = 0 ; i < SYS_ARRAY_NUM_COLS ; i++){
                steps[row][i] = 0;
            }
        }
    }

    static <E> void write_channel_intel(WatchFIFO<E> q, E data) throws InterruptedException{
        //System.err.printf("-> write %s\n", s);
        q.put(data);
        //System.err.printf("<- wrote %s\n", s);
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

    static <E> E read_channel_intel(WatchFIFO<E> q) throws InterruptedException{
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

        public void run(){
            try {
                boolean first = true;
                boolean cont = true;
                boolean flush = false;
                int y = 0;
                int x = 0;
                int reuse = 0;
                int yblock = 0;
                int ybase = 0;
                int index = 0;
                while(cont){
                    //int index = (yblock * MATRIX_A_BLOCK_HEIGHT + y) * mat_a_num_vectors_in_row + x ;
                    VecFloat load = flush ? VECTOR_ZERO : A[index];
                    //System.err.printf("Load %d %d %d %d\n", y, x, reuse, yblock);
                    ChannelAData write = new ChannelAData(load, !first && x == 0);
                    write_channel_intel(row_feed_chain_border, write);
                    if(y == MATRIX_A_BLOCK_HEIGHT - 1){
                        y = 0;
                        first = false;
                        if(flush && x == 2){
                            cont = false;
                        } else if(x == mat_a_num_vectors_in_row -1){
                            x = 0;
                            if(reuse == mat_b_num_blocks_in_row){
                                reuse = 0;
                                if(yblock == mat_a_num_blocks_in_col - 1){
                                    flush = true;
                                } else {
                                    ybase = index + 1;
                                    yblock++;
                                }
                            } else {
                                reuse++;
                            }
                        } else {
                           x++;
                        }
                        index = ybase + x;
                    } else {
                        y++;
                        index += mat_a_num_vectors_in_row;
                    }
                }



            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! LoadMatA" + e.getMessage());
                System.exit(0);
            }

        }

        public void runSingle(){
            try {
                boolean first = true;
                for (int yblock = 0; yblock < mat_a_num_blocks_in_col + 1; yblock++) {
                    for (int reuse = 0; reuse < (yblock < mat_a_num_blocks_in_col ? mat_b_num_blocks_in_row : 1); reuse++) {
                        for(int x = 0 ; x < (yblock < mat_a_num_blocks_in_col ? mat_a_num_vectors_in_row : 2) ; x++){
                            for(int y = 0 ; y < MATRIX_A_BLOCK_HEIGHT; y++){
                                int index = (yblock * MATRIX_A_BLOCK_HEIGHT + y) * mat_a_num_vectors_in_row + x ;
                                VecFloat load = (yblock < mat_a_num_blocks_in_col ? A[index] : VECTOR_ZERO);
                                ChannelAData write = new ChannelAData(load, !first && x == 0);
                                write_channel_intel(row_feed_chain_border, write);
                            }
                            first = false;
                        }
                        //System.err.printf(" Wrote matrix A BLOCK %d \n", yblock  );
                    }
                }


            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! LoadMatA" + e.getMessage());
                System.exit(0);
            }

        }

        public void runSimple() {
            try {
                boolean first = true;
                for (int yblock = 0; yblock < mat_a_num_blocks_in_col; yblock++) {
                    for (int reuse = 0; reuse < mat_b_num_blocks_in_row; reuse++) {
                        for(int x = 0 ; x < mat_a_num_vectors_in_row ; x++){
                            for(int y = 0 ; y < MATRIX_A_BLOCK_HEIGHT; y++){
                                int index = (yblock * MATRIX_A_BLOCK_HEIGHT + y) * mat_a_num_vectors_in_row + x ;
                                ChannelAData write = new ChannelAData(A[index], !first && x == 0);
                                write_channel_intel(row_feed_chain_border, write);
                            }
                            first = false;
                        }
                            //System.err.printf(" Wrote matrix A BLOCK %d \n", yblock  );
                    }
                }
                // flush last block
                for(int row = 0 ; row <  MATRIX_A_BLOCK_HEIGHT ; row++) {
                    ChannelAData write = new ChannelAData(VECTOR_ZERO, true);
                    write_channel_intel(row_feed_chain_border,write, "row_feed_chain[0]" );
                }
                // Buffer will not flush without new elements
                for(int row = 0 ; row < MATRIX_A_BLOCK_HEIGHT ; row++) {
                    ChannelAData write = new ChannelAData(VECTOR_ZERO, false);
                    write_channel_intel(row_feed_chain_border,write, "row_feed_chain[0]" );
                }
                System.err.println("DONE!!! LOAD MAT A!");

            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! LoadMatA" + e.getMessage());
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

        public void run(){
            final int mat_b_num_vectors_in_col = mat_a_num_vectors_in_row;
            try {
                int reuse = 0;
                boolean flush = false;
                boolean cont = true;
                int xblock = 0;
                int y = 0;
                int x = 0;
                int index = 0;
                int xbase = 0;
                while(cont){

                    //int index = (xblock * MATRIX_B_BLOCK_WIDTH + x) * mat_b_num_vectors_in_col + y;
                    VecFloat load = flush ? VECTOR_ZERO : B[index] ;
                    write_channel_intel(col_feed_chain_border, load);

                    if(x == MATRIX_B_BLOCK_WIDTH - 1){
                        x = 0;
                        if(flush && y == 2){
                            cont = false;
                        } else if(y == mat_a_num_vectors_in_row-1){
                            y = 0;
                            if(xblock == mat_b_num_blocks_in_row - 1){
                                xblock = 0;
                                if(reuse == mat_a_num_blocks_in_col - 1){
                                    flush=true;
                                } else {
                                    index = 0;
                                    xbase = 0;
                                    reuse++;
                                }
                            } else {
                                xblock++;
                            }
                        } else {
                            y++;
                        }
                        index = xbase + y;
                    } else {
                        x++;
                        index+=mat_b_num_vectors_in_col;
                    }
                }
            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! LoadMatB" + e.getMessage());
                System.exit(0);
            }
        }

        public void runSingle() {
            final int mat_b_num_vectors_in_col = mat_a_num_vectors_in_row;
            try {
                for (int reuse = 0; reuse < mat_a_num_blocks_in_col; reuse++) {
                    for (int xblock = 0; xblock < mat_b_num_blocks_in_row; xblock++) {
                        int nrY = mat_a_num_vectors_in_row;
                        if(reuse == mat_a_num_blocks_in_col - 1 && xblock == mat_b_num_blocks_in_row - 1){
                            nrY +=2;
                        }
                        for (int y = 0; y < nrY; y++) {
                            for (int x = 0; x < MATRIX_B_BLOCK_WIDTH; x++) {
                                int index = (xblock * MATRIX_B_BLOCK_WIDTH + x) * mat_b_num_vectors_in_col + y;
                                VecFloat load = y < mat_a_num_vectors_in_row ? B[index] : VECTOR_ZERO;
                                write_channel_intel(col_feed_chain_border, load);
                            }
                        }
                    }
                }

            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! LoadMatB" + e.getMessage());
                System.exit(0);
            }
        }

        public void runSimple() {
            final int mat_b_num_vectors_in_col = mat_a_num_vectors_in_row;
            try {
                for (int reuse = 0; reuse < mat_a_num_blocks_in_col; reuse++) {
                    for (int xblock = 0; xblock < mat_b_num_blocks_in_row; xblock++) {
                        int nrY = mat_a_num_vectors_in_row;
                        if(reuse == mat_a_num_blocks_in_col - 1 && xblock == mat_b_num_blocks_in_row){
                            nrY +=2;
                        }
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
                    write_channel_intel(col_feed_chain_border, VECTOR_ZERO);
                }
            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! LoadMatB" + e.getMessage());
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

                int freq = (SYS_ARRAY_NUM_ROWS - 1) - row;
                int count = freq;
                while(true) {
                    ChannelAData read;
                    if(row == 0 ){
                        read = read_channel_intel(row_feed_chain_border);
                    } else {
                        read = read_channel_intel(row_feed_chain[row-1]);
                    }
                    //System.err.printf("Read from feed row chain %d\n", row);
                    if(count == 0) {
                        write_channel_intel(row_feed_to_buf[row], read);
                        count = freq;
                    } else {
                        write_channel_intel(row_feed_chain[row], read);
                        count--;
                    }
                }

            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! FeedMatA" + e.getMessage());
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

                int freq = (SYS_ARRAY_NUM_COLS - 1) - col;
                int count = freq;
                while(true) {
                    VecFloat read;
                    if(col == 0) {
                        read = read_channel_intel(col_feed_chain_border);
                    } else {
                        read = read_channel_intel(col_feed_chain[col-1]);
                    }
                    if(count == 0) {
                        write_channel_intel(col_feed_to_buf[col], read);
                        count = freq;
                    } else {
                        write_channel_intel(col_feed_chain[col], read);
                        count--;
                    }
                }

            } catch(InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! FeedMatB" + e.getMessage());
                System.exit(0);
            }
        }
    }

    static class Buf_mat_a_kernel extends Thread{

        final int row;
        final VecFloat[] bufShift;
        final VecFloat[] backShift;
        final VecFloat[][] buf;
        final boolean[] new_row_col;

        Buf_mat_a_kernel(int row){
            this.row = row;
            buf = new VecFloat[2][];
            bufShift = new VecFloat[INTERLEAVED];
            for(int r = 0 ; r < INTERLEAVED ; r++) {
                bufShift[r] = VECTOR_ZERO;
            }
            backShift = new VecFloat[INTERLEAVED];
            for(int r = 0 ; r < INTERLEAVED ; r++) {
                backShift[r] = VECTOR_ZERO;
            }
            for(int b = 0 ; b < 2 ; b++){
                buf[b] = new VecFloat[INTERLEAVED];
                for(int r = 0 ; r < INTERLEAVED ; r++) {
                    buf[b][r] = VECTOR_ZERO;
                }
            }
            new_row_col = new boolean[2];
            new_row_col[0] = new_row_col[1] = false;
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
                    ChannelAData write = new ChannelAData(buf[buf_to_read][row_to_read_write],new_row_col[buf_to_read]);
                    write_channel_intel(ch_data_a_border[row],write );
                    if(reuse_cnt == INTERLEAVED - 1) {
                        reuse_cnt = 0;
                        // load once every COLS_INTERLEAVED steps
                        ChannelAData read = read_channel_intel(row_feed_to_buf[row]);
                        int buf_to_write = 1 - buf_to_read;
                        buf[buf_to_write][row_to_read_write] = read.data;
                        if(row_to_read_write == 0) {
                            new_row_col[buf_to_write] = read.new_row_col_pair;
                        }
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
                e.printStackTrace();
                System.err.println("Interrupted!! Buf_mat_a_kernel" + e.getMessage());
                System.exit(0);
            }
        }
    }


    static class Buf_mat_b_kernel extends Thread{

        final int col;
        final VecFloat[][] buf;
        final VecFloat[] bufShift;
        final VecFloat[] backShift;

        Buf_mat_b_kernel(int col){
            this.col = col;
            buf = new VecFloat[2][];
            for(int b = 0 ; b < 2 ; b++){
                buf[b] = new VecFloat[INTERLEAVED];
                for(int r = 0 ; r < INTERLEAVED ; r++) {
                    buf[b][r] = VECTOR_ZERO;
                }
            }
            bufShift = new VecFloat[INTERLEAVED];
            for(int r = 0 ; r < INTERLEAVED ; r++) {
                bufShift[r] = VECTOR_ZERO;
            }
            backShift = new VecFloat[INTERLEAVED-1];
            for(int r = 0 ; r < INTERLEAVED -1 ; r++) {
                backShift[r] = VECTOR_ZERO;
            }
        }


        public void run(){
            try{
                // We obtain a new value every ROWS_INTERLEAVED steps
                // Each value is also reused ROWS_INTERLEAVED steps
                // meaning new values come in at exactly the right rate (assuming ROWS_INTERLEAVED = COLS_INTERLEAVED)
                int reuse_cnt_col_to_write = 0;
                int col_to_read = 0;
                while (true) {
                    VecFloat first = bufShift[0];
                    VecFloat firstBack = backShift[0];
                    write_channel_intel(ch_data_b_border[col],first);
                    boolean flip = reuse_cnt_col_to_write == INTERLEAVED -1;
                    boolean read = col_to_read == INTERLEAVED - 1;
                    String s = flip ? "flip" : "noflip";
                    String s2 = read ? "read" : "noread";
                    VecFloat newRead ;
                    if(read){
                        newRead = read_channel_intel(col_feed_to_buf[col]);
                    } else {
                        newRead = VECTOR_ZERO;
                    }
                    //System.err.printf("Buf %d %d %f %f %s %s\n", col_to_read, reuse_cnt_col_to_write, first.vals[0], newRead.vals[0], s ,s2);
                    for(int i = 0; i < INTERLEAVED - 1 ; i++){
                        bufShift[i] = bufShift[i + 1];
                    }
                    bufShift[INTERLEAVED -1] = flip ? (read? newRead : firstBack) : first;

                    for(int i = 0; i < INTERLEAVED - 2 ; i++){
                        backShift[i] = backShift[i + 1];
                    }
                    backShift[INTERLEAVED - 2] = flip ? VECTOR_ZERO : (read ? newRead : firstBack);

                    if(read){
                        col_to_read = 0;

                        if(flip){
                            reuse_cnt_col_to_write = 0;
                        } else {
                            reuse_cnt_col_to_write++;
                        }
                    } else {
                        col_to_read++;
                    }

                }
            } catch (InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! Buf_mat_b_kernel" + e.getMessage());
                System.exit(0);
            }
        }

        public void runSimple(){
            try{
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
                        VecFloat read = read_channel_intel(col_feed_to_buf[col]);
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
            } catch (InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! Buf_mat_b_kernel" + e.getMessage());
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

                    ChannelAData read_A;
                    if(col == 0) {
                        read_A = read_channel_intel(ch_data_a_border[row]);
                    } else {
                        read_A = read_channel_intel(ch_data_a[row][col-1]);
                    }
                    VecFloat a_data = read_A.data;
                    boolean new_col_row_pair = read_A.new_row_col_pair;

                    if (col < (SYS_ARRAY_NUM_COLS-1)) {
                        write_channel_intel(ch_data_a[row][col], read_A);
                    }

                    VecFloat b_data;
                    if(row == 0){
                        b_data = read_channel_intel(ch_data_b_border[col]);
                    } else {
                        b_data = read_channel_intel(ch_data_b[row-1][col]);
                    }


                    if (row < (SYS_ARRAY_NUM_ROWS-1)) {
                        write_channel_intel(ch_data_b[row][col], b_data);
                    }

                    if(new_col_row_pair) {
                        write_channel_intel(ch_data_c[row][col],interleave_shift[NR_INTERLEAVED-1]);
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
                }
            } catch(InterruptedException e){
                e.printStackTrace();
                System.err.println("Interrupted!! PE_Kernel" + e.getMessage());
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
            } catch (InterruptedException e){
                e.printStackTrace();
                System.err.println("Interrupted!! Drain_C " + e.getMessage());
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
                    float in = read_channel_intel(ch_drain_c_border[col]);
                    float[] prev_node_data_in = new float[SYS_ARRAY_NUM_COLS];
                    if(col != SYS_ARRAY_NUM_COLS - 1) {
                        prev_node_data_in = read_channel_intel(col_c_chain[col]).vals;
                    }

                    float[] out = new float[SYS_ARRAY_NUM_COLS];

                    for (int i = 0; i < SYS_ARRAY_NUM_COLS - 1; i++) {
                        out[i] = prev_node_data_in[i+1];
                    }

                    out[SYS_ARRAY_NUM_COLS-1] = in;
                    if(col == 0){
                        write_channel_intel(col_c_chain_border, new VecFloat(out));
                    } else {
                        write_channel_intel(col_c_chain[col-1], new VecFloat(out));
                    }

                }
            } catch (Exception e){
                e.printStackTrace();
                System.err.println("Interrupted!! Drain_C_cols " + e.getMessage());
                System.exit(0);
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
            try {
                for(int yblock = 0 ; yblock < nrYBlocks ; yblock++){
                    for(int xblock = 0; xblock < nrXBlocks ; xblock++){
                        for(int ylocal = 0 ; ylocal < MATRIX_A_BLOCK_HEIGHT ; ylocal++) {
                            for (int xlocal = 0; xlocal < INTERLEAVED; xlocal++) {
                                int index = ((yblock * MATRIX_A_BLOCK_HEIGHT + ylocal) * nrXBlocks + xblock) * INTERLEAVED  + xlocal;
                                VecFloat dataIn = read_channel_intel(col_c_chain_border);
                                toMem[index] = dataIn;
                            }
                        }
                    }
                }
            }  catch (InterruptedException e) {
                e.printStackTrace();
                System.err.println("Interrupted!! Drain_to_mem " + e.getMessage());
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

        new Drain_to_mem(c,widthb / MATRIX_B_BLOCK_WIDTH, heighta / MATRIX_A_BLOCK_HEIGHT).run();

    }

    public static float[][] matrixMulSystolic(float[][] a, float[][] b, int widtha, int heighta, int widthb){
        if(heighta % MATRIX_A_BLOCK_HEIGHT != 0 || widthb % MATRIX_B_BLOCK_WIDTH !=0){
            System.err.println("WRONG DIMENSIONS!");
            System.exit(0);
        }
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


    public static float[][] testMat(int size) {
        float[][] res = new float[size][];
        int z = 0;
        for(int row = 0 ; row < size ; row++ ){
            res[row] = new float[size];
            for(int col = 0 ; col < size ; col++){
                /*if(col == row) {
                    res[row][col] = (float)Math.sqrt(row + 1);
                } else {
                    res[row][col] = 0;
                }
                */
                res[row][col] = z;
                z++;
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
            System.err.println(row(col_c_chain));

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
        int width = Math.max(MATRIX_A_BLOCK_HEIGHT,MATRIX_B_BLOCK_WIDTH) * 1 ;
        //new WatchEm().start();;
        float[][] ident = testMat(width);
        checkem(ident,ident,width,width,width, 0.001f);
        System.out.println("DONE");
//        try {
//            Thread.sleep(100000);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//        System.exit(0);
    }

}
