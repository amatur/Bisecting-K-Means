
import java.util.Random;

/**
 *
 * @author Hera
 */
public class Main {
    
    public static void main (String[] args) {
        
        Random random = new Random();
        
        for (int i = 0; i < 50; i++)
            System.out.println((0+random.nextGaussian()*5) + " " + (0+random.nextGaussian()*5));
        for (int i = 0; i < 50; i++)
            System.out.println((50+random.nextGaussian()*5) + " " + (50+random.nextGaussian()*5));
        
    }
    
}
