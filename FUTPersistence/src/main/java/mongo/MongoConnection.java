package mongo;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoDatabase;
import configuration.LoadXmlConf;
import configuration.MongoConfig;

abstract class MongoConnection implements AutoCloseable{
    private static final String MONGO_CONFIG = "mongoConfig";
    private static final MongoConfig  mongoConfig = new LoadXmlConf().getConfigIstance(MONGO_CONFIG);
    protected static final MongoClient mongoClient;
    protected static final MongoDatabase db;

    static {
        //open connection
        mongoClient = MongoClients.create("mongodb://"+mongoConfig.mongoIp+":"+mongoConfig.mongoPort);
        db = mongoClient.getDatabase(mongoConfig.dbName);
    }

    @Override
    public void close(){
        mongoClient.close();
    }
}
