import bean.Player;
import bean.User;
import com.mongodb.client.*;

import java.text.SimpleDateFormat;
import java.util.*;

import org.bson.Document;

import static com.mongodb.client.model.Filters.*;


public class ProvaQuery {
    public void user_registration(String username, String first_name, String last_name, String country, String password){
        //open connection
        MongoClient mongoClient = MongoClients.create("mongodb://localhost:27017");
        MongoDatabase db = mongoClient.getDatabase("futdb");
        MongoCollection<Document> myColl = db.getCollection("users");
        //query
        SimpleDateFormat df = new SimpleDateFormat("dd/MM/yyyy");
        String date = df.format(new Date());
        Document user = new Document("username", username)
                .append("first_name", first_name)
                .append("last_name", last_name)
                .append("country", country)
                .append("join_date", date)
                .append("password", password);
        myColl.insertOne(user);
        //close connection
        mongoClient.close();
    }

    public User show_profile_information(String username){
        //open connection
        MongoClient mongoClient = MongoClients.create("mongodb://localhost:27017");
        MongoDatabase db = mongoClient.getDatabase("futdb");
        MongoCollection<Document> myColl = db.getCollection("users");
        //query
        Document doc = myColl.find(eq("username",username)).first();
        //close connection
        mongoClient.close();

        //User newUser = new User(doc.get("first_name").toString(),//da completare con gli altri attributi);
        return new User();
    }

    public Integer count_document_in_collection(String collection){
        //open connection
        MongoClient mongoClient = MongoClients.create("mongodb://localhost:27017");
        MongoDatabase db = mongoClient.getDatabase("futdb");
        MongoCollection<Document> myColl = db.getCollection(collection);
        //query
        Integer i = Math.toIntExact(myColl.countDocuments());
        //close connection
        mongoClient.close();
        return i;
    }

    public Player show_player_information(Integer id){
        //open connection
        MongoClient mongoClient = MongoClients.create("mongodb://localhost:27017");
        MongoDatabase db = mongoClient.getDatabase("futdb");
        MongoCollection<Document> myColl = db.getCollection("players");
        //query
        Document doc = myColl.find(eq("futbin_id",id)).first();
        //close connection
        mongoClient.close();

        //User newUser = new User(doc.get("first_name").toString(),//da completare con gli altri attributi);
        System.out.println(doc.toJson());
        return new Player();
    }

    public static void main(String[] args){
        ProvaQuery m = new ProvaQuery();
        m.show_player_information(1);
    }
}




