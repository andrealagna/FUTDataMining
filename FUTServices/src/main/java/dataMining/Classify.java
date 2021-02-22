package dataMining;

import bean.Comment;

import com.github.pemistahl.lingua.api.Language;
import com.github.pemistahl.lingua.api.LanguageDetector;
import com.github.pemistahl.lingua.api.LanguageDetectorBuilder;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.core.*;
import weka.core.SerializationHelper;

import java.util.ArrayList;

public class Classify {

    public ArrayList<Comment> cleanComments(ArrayList<Comment> comments) throws Exception{
        ArrayList<Comment> c = new ArrayList<>();
        String cleanedComment;
        final LanguageDetector detector = LanguageDetectorBuilder.fromLanguages(Language.ITALIAN, Language.ENGLISH).build();
        for(int i = 0; i < comments.size(); i++) {
            cleanedComment = comments.get(i).getText();
            cleanedComment.replaceAll("\\s+", " ");
            cleanedComment.replaceAll("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]"," ");
            final Language detectedLanguage = detector.detectLanguageOf(cleanedComment);
            if(detectedLanguage.toString().equals("ENGLISH")) {
                Comment newComment = new Comment(comments.get(i).getId(),comments.get(i).getDate(),cleanedComment,comments.get(i).getAuthor());
                c.add(newComment);
            }
        }
        return c;
    }



    public ArrayList<Comment> labelComments(ArrayList<Comment> c) throws Exception{

        ArrayList<Comment> comments = cleanComments(c);

        ArrayList<String> classes;
        classes = new ArrayList<String>();
        classes.add("negative");
        classes.add("neutral");
        classes.add("positive");

        Attribute clas;
        Attribute text;
        text = new Attribute("text", true);
        clas = new Attribute("class", classes);

        ArrayList<Attribute> attributes;
        attributes = new ArrayList<Attribute>();
        attributes.add(text);
        attributes.add(clas);

        Instances dataset;
        Instance instance;
        dataset = new Instances("comments", attributes, comments.size());
        dataset.setClassIndex(dataset.numAttributes()-1);

        for(int i = 0; i < comments.size(); i++) {
            double[] value = new double[dataset.numAttributes()];
            value[0] = dataset.attribute(0).addStringValue(comments.get(i).getText());
            instance = new DenseInstance(1.0, value);
            dataset.add(instance);
            dataset.instance(i).setClassMissing();
        }

        NaiveBayesMultinomialText classifier;
        classifier = (NaiveBayesMultinomialText) SerializationHelper.read(
                "../FUTServices/src/main/resources/NaiveBayesMultinomialText_sliding_6.model");

        for(int i = 0; i < comments.size(); i++) {
            double predicted;
            dataset.instance(i).setClassValue(classifier.classifyInstance(dataset.instance(i)));
            predicted = dataset.instance(i).classValue();
            if(predicted == 0) { //0: negative. 1: neutral. 2:positive
                comments.get(i).setSentiment("negative");
            } else if (predicted == 2) {
                comments.get(i).setSentiment("positive");
            }else{
                comments.get(i).setSentiment("neutral");
            }
        }
        return comments;
    }
}


