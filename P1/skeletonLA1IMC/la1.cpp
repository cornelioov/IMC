//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool tflag= 0, Tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, sflag = 0, wflag = 0, pflag = 0;
    char *tvalue = NULL, *Tvalue = NULL, *wvalue = NULL;
    int c;

    opterr = 0;

    /////////////////////////////////////////////////////
    int ivalue = 0, lvalue = 0, hvalue = 0;
    float evalue = 0.0, mvalue = 0.0;
    /////////////////////////////////////////////////////

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:s:w:p")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'i':
                iflag = true;
                ivalue = atoi(optarg);
                break;
            case 'l':
                lflag = true;
                lvalue = atoi(optarg);
                break;
            case 'h':
                hflag = true;
                hvalue = atoi(optarg);
                break;
            case 'e':
                eflag = true;
                evalue = atof(optarg);
                break;
            case 'm':
                mflag = true;
                mvalue = atof(optarg);
                break;
            case 's':
                sflag = true;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if(!tflag)
    {
        cerr << "ERROR: Argument -t is mandatory" << endl;
        return EXIT_FAILURE;
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        if(!iflag)
        {
            ivalue = 1000;
        }

        if(!lflag)
        {
            lvalue = 1;
        }

        if(!hflag)
        {
            hvalue = 5;
        }

        if(!eflag)
        {
            evalue = 0.1;
        }

        if(!mflag)
        {
            mvalue = 0.9;
        }

        if(!Tflag)
        {
            Tvalue = tvalue;
        }

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
    	int iterations = ivalue; // This should be corrected

        // Read training and test data: call to util::readData(...)
    	Dataset * trainDataset = readData(tvalue); // This should be corrected
        Dataset * testDataset;
    	if(!Tflag)
        {
            testDataset = trainDataset;
        }
        else
        {
            testDataset = readData(Tvalue);
        }

        if(sflag)
        {
            // Escalamos los datos de entrada
            double *minTrain = minDatasetInputs(trainDataset);
            double *maxTrain = maxDatasetInputs(trainDataset);
            minMaxScalerDataSetInputs(trainDataset,-1,1,minTrain,maxTrain);
            minMaxScalerDataSetInputs(testDataset,-1,1,minTrain,maxTrain);

            //Escalamos los datos de salida
            double minTest = minDatasetOutputs(testDataset);
            double maxTest = maxDatasetOutputs(testDataset);
            minMaxScalerDataSetOutputs(trainDataset,0,1,minTest,maxTest);
            minMaxScalerDataSetOutputs(testDataset,0,1,minTest,maxTest);
        }

        // Initialize topology vector
    	int layers=lvalue; // This should be corrected
    	int * topology=(int *)(lvalue + 2); // This should be corrected

        topology[0] = trainDataset->nOfInputs;
        topology[layers+1] = trainDataset->nOfOutputs;

        for(int i=1; i<layers+1; i++)
        {
            topology[i] = hvalue;
        }

        // Initialize the network using the topology vector
        mlp.initialize(layers+2,topology);


        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = DBL_MAX;
        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;
        
        // Obtain training and test averages and standard deviations
        for(int i=0; i<5; i++)
        {
            averageTestError += testErrors[i];
            averageTrainError += trainErrors[i];
        }
        averageTestError = averageTestError/5;
        averageTrainError = averageTrainError/5;

        for(int i=0; i<5; i++)
        {
            stdTestError += pow(testErrors[i] - averageTestError, 2);
            stdTrainError += pow(trainErrors[i] - averageTrainError, 2);
        }
        stdTestError = sqrt(stdTestError/5);
        stdTrainError = sqrt(stdTrainError/5);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to util::readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }

    
}

