import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class IrisApp {
    public static void main(String[] args) throws Exception {
        double learninRate=0.001;
        int batchSize=1;
        int inputSize=4; int outputSize=3;
        int classIndex=4;
        int numHidden=10;
        MultiLayerNetwork model;
        int nEpochs=500;
        InMemoryStatsStorage inMemoryStatsStorage;

        System.out.println("Création du modèle");
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learninRate))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID).build())
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHidden)
                        .nOut(outputSize)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();
        model=new MultiLayerNetwork(configuration);
        model.init();
       //Démarage du serveur de monitoring du processus d'apprentissage
        UIServer uiServer=UIServer.getInstance();
        inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));
        // System.out.println(configuration.toJson());
        System.out.println("Entrainement du modèle");
        File fileTrain=new ClassPathResource("iris-train.csv").getFile();
        //Lire les données du fichier csv avec séparateur par défaut(virgule)
        RecordReader recordReaderTrain=new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        DataSetIterator dataSetIteratorTrain=
                new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,outputSize);

/*
        while(dataSetIteratorTrain.hasNext()){
            System.out.println("------------------------------");
            DataSet dataSet=dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }
        */

        //entrainement

       for (int i = 0; i <nEpochs ; i++) {
            model.fit(dataSetIteratorTrain);
        }

        System.out.println("Model Evaluation");
        File fileTest=new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest=new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest=
                new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,outputSize);

        Evaluation evaluation=new Evaluation(outputSize);

        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray TargetLabels=dataSet.getLabels();
            INDArray predictedLabels=model.output(features);
            evaluation.eval(predictedLabels,TargetLabels);
        }
        System.out.println(evaluation.stats());
        ModelSerializer.writeModel(model,"irisModel.zip",true);




    }
}