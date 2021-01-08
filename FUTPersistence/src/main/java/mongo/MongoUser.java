package mongo;

import bean.*;
import com.mongodb.ReadConcern;
import com.mongodb.client.*;
import com.mongodb.client.model.Aggregates;
import org.bson.Document;
import org.bson.conversions.Bson;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.function.Consumer;
import java.util.regex.Pattern;

import static com.mongodb.client.model.Accumulators.avg;
import static com.mongodb.client.model.Accumulators.sum;
import static com.mongodb.client.model.Aggregates.*;
import static com.mongodb.client.model.Filters.*;
import static com.mongodb.client.model.Projections.*;
import static com.mongodb.client.model.Projections.include;
import static com.mongodb.client.model.Sorts.descending;
import static com.mongodb.client.model.Updates.inc;

public class MongoUser extends MongoConnection{
    private MongoCollection<Document> myColl;

    public String add(String firstName, String lastName, String username, String country, String joinDate, String password){
        myColl = db.getCollection("users");
        Document user = new Document("username", username)
                .append("first_name", firstName)
                .append("last_name", lastName)
                .append("country", country)
                .append("join_date", joinDate)
                .append("password", password);
        myColl.insertOne(user);

        return user.getObjectId("_id").toString();
    }

    public User getUser(String username){
        myColl = db.getCollection("users");
        //query
        Document doc = myColl.find(eq("username",username)).first();
        return composeUser(doc);

    }

    public ArrayList<User> findUsers(String toFind) {
        myColl = db.getCollection("users");
        ArrayList<User> users = new ArrayList<>();

        try (MongoCursor<Document> cursor = myColl.find(regex("username",".*" + Pattern.quote(toFind) + ".*", "-i")).iterator())
        {
            while (cursor.hasNext())
            {
                Document userDoc = cursor.next();
                User user = composeUser(userDoc);
                users.add(user);
            }
        } catch (Exception e){
            e.printStackTrace();
        }
        return users;
    }

    private User composeUser(Document doc){
        //no user found
        if (doc == null)
            return null;

        SimpleDateFormat df = new SimpleDateFormat("dd/MM/yyyy");
        Date date = null;
        try {
            date = df.parse(doc.get("join_date").toString());
        } catch (ParseException e) {
            e.printStackTrace();
        }

        //ricostruisco le squadre dell'utente
        ArrayList<Document> squadsDoc = (ArrayList)doc.get("squads");
        //inserire se non ha squadra, inizializzatlo vuoto
        ArrayList<Squad> s = new ArrayList<>();
        if(squadsDoc == null){
            s = null;
        }
        else {
            for (Document squad : squadsDoc) {
                HashMap<String, Player> pos = new HashMap<>();
                Map<String, String> map = (Map) squad.get("players");
                Iterator iterator = map.keySet().iterator();
                MongoPlayerCard mongoPlayerCard = new MongoPlayerCard();
                while (iterator.hasNext()) {
                    String key = iterator.next().toString();
                    String value = map.get(key);

                    Player x = mongoPlayerCard.findById(Integer.parseInt(value));
                    if (x == null) //utente non caricato nel sistema
                        continue;
                    pos.put(key, x);
                }
                try {
                    df = new SimpleDateFormat("dd/MM/yyyy");
                    date = df.parse(squad.get("date").toString());
                } catch (ParseException e) {
                    e.printStackTrace();
                }

                Squad sq = new Squad(squad.get("name").toString(), squad.get("module").toString(),
                        date, pos);
                s.add(sq);
            }
        }

        User newUser = new User(doc.get("username").toString(), doc.get("first_name").toString(),
                doc.get("last_name").toString(), doc.get("_id").toString(),
                doc.get("country").toString(), date, doc.get("password").toString(), s, Integer.parseInt(doc.get("score").toString()));
        return newUser;
    }

    public Integer countElement(String collection){
        myColl = db.getCollection("users");
        return Math.toIntExact(myColl.countDocuments());
    }

    public void updateScore (String userId, int points){
        myColl = db.getCollection("users");
        //query
        myColl.updateOne(eq("_id", userId), inc("score", points));
    }

    public int getScore (String userId){
        myColl = db.getCollection("users");
        //query
        Document doc = myColl.find(eq("_id", userId)).first();
        assert doc != null;
        return (Integer) doc.get("score");
    }
/*
    public void analyticsOne(String nationality){
        myColl = db.getCollection("users").withReadConcern(ReadConcern.AVAILABLE);

        Consumer<Document> printDocuments = doc -> {System.out.println(doc.toJson());};

        Bson unwindSquad = unwind("$squads");
        Bson convertDate = addFields("squads.date");

        Bson subtractDate =
        Bson matchNationality = match(and(eq("nationality", nationality), ne("league", "Icons")));
        Bson groupLeague = group("$league",
                sum("numPlayers", 1),
                avg("paceAvg", "$pace"),
                avg("dribblingAvg", "$dribbling"),
                avg("shootingAvg", "$shooting"),
                avg("passingAvg", "$passing"),
                avg("defendingAvg", "$defending"),
                avg("physicalityAvg", "$physicality")
        );
        Bson sort = sort(descending("numPlayers"));
        Bson limit = limit(3);
        Bson project = project(fields(excludeId(),
                computed("league", "$_id"),
                include("numPlayers"),
                include("paceAvg"),
                include("dribblingAvg"),
                include("shootingAvg"),
                include("passingAvg"),
                include("defendingAvg"),
                include("physicalityAvg")
                )
        );

        Aggregates.project(fields(computed()))

        myColl.aggregate(Arrays.asList(matchNationality, groupLeague, sort, limit, project)).forEach(printDocuments);
    }

 */



    public static void main(String[] args){

        //System.out.println(m.findById(10));
    }
}




