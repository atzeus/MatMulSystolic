import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Created by atze on 2-10-17.
 */
public class WatchFIFO<E> {

    final BlockingQueue<E> q;
    boolean offering;
    boolean taking;
    int passed;


    WatchFIFO(int size) {
        this.q = new ArrayBlockingQueue<E>(size);
        this.offering = false;
        this.taking = false;
        passed = 0;
    }

    void put(E val) throws InterruptedException{
        offering = true;
        q.put(val);
        offering = false;
        passed++;
    }

    E take()  throws InterruptedException{
        taking = true;
        E ret = q.take();
        taking = false;
        return ret;
    }


}
