package dataMining;

import bean.Comment;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.*;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class Classify {

    public ArrayList<Comment> labelComments(ArrayList<Comment> c) throws Exception{

        ArrayList<String> classLabels;
        classLabels = new ArrayList<String>();
        classLabels.add("negative");
        classLabels.add("neutral");
        classLabels.add("positive");

        //Attributes
        Attribute clas;
        Attribute text;
        text = new Attribute("text", true);
        clas = new Attribute("class", classLabels);

        //List of attributes
        ArrayList<Attribute> attributes;
        attributes = new ArrayList<Attribute>();
        attributes.add(text);
        attributes.add(clas);

        //Dataset
        Instances dataset;
        Instance inst;
        dataset = new Instances("comments", attributes, c.size());
        dataset.setClassIndex(dataset.numAttributes()-1);

        //Creation of the instances of the dataset and set the class as missing (unlabeled)
        for(int i = 0; i < c.size(); i++) {
            double[] value = new double[dataset.numAttributes()];
            value[0] = dataset.attribute(0).addStringValue(c.get(i).getText());
            inst = new DenseInstance(1.0, value);
            dataset.add(inst);
            dataset.instance(i).setClassMissing();
        }

        FilteredClassifier classifier;
        classifier = (FilteredClassifier) SerializationHelper.read("C:\\Users\\Andrea Lagna\\IdeaProjects\\FUTDM\\FUTServices\\src\\main\\resources\\SMO.model");

        for(int i = 0; i < c.size(); i++) {
            double predicted;
            dataset.instance(i).setClassValue(classifier.classifyInstance(dataset.instance(i)));
            predicted = dataset.instance(i).classValue();
            if(predicted == 0) { //Index of the class label array -> 0: negative. 1: neutral. 2:positive
                c.get(i).setSentiment("negative");
            } else if (predicted == 2) {
                c.get(i).setSentiment("positive");
            }else{
                c.get(i).setSentiment("neutral");
            }
        }
        return c;
    }
}