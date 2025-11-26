package ulpgc.shared;

import com.hazelcast.config.Config;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;

public class SimpleHazelcastCluster {
    public static void main(String[] args) {
        Config config = new Config();
        config.setClusterName("test-cluster"); // Nombre del clúster de Hazelcast
        HazelcastInstance hz = Hazelcast.newHazelcastInstance(config);
        System.out.println("Nodo conectado al clúster: " + hz.getCluster().getLocalMember().getUuid());
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        hz.shutdown();
    }
}
