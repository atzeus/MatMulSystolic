import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Created by atze on 2-10-17.
 */
public class WatchFIFO<E> {

    final BlockingQueue<E> q;
    Thread read;
    Thread write;
    boolean offering;
    boolean taking;
    int passed;


    WatchFIFO(int size) {
        this.q = new ArrayBlockingQueue<E>(size);
        this.offering = false;
        this.taking = false;
        passed = 0;
        read = null;
        write = null;
    }

    void put(E val) throws InterruptedException{
        if(write == null) {
            write = Thread.currentThread();
        } else if (write != Thread.currentThread()){
            throw new Error("Not the same thread writing!");
        }
        offering = true;
        q.put(val);
        offering = false;
        passed++;
    }

    E take()  throws InterruptedException{
        if(read == null) {
            read = Thread.currentThread();
        } else if (read != Thread.currentThread()){
            throw new Error("Not the same thread reading!");
        }
        taking = true;
        E ret = q.take();
        taking = false;
        return ret;
    }


}
