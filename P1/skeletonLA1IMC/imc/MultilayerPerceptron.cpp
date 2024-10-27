/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	this->nOfLayers = 1;
	this->eta = 0.1;
	this->mu = 0.9;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vector containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	
	this->nOfLayers = nl;
	this->layers = new Layer[nl];
	
	for(int i=0; i< this->nOfLayers; i++)
	{
		this->layers[i].nOfNeurons = npl[i];
		this->layers[i].neurons = new Neuron[npl[i]];
	}

	for (int i = 0; i < nl; i++) 
	{
		for (int j = 0; j < npl[i]; j++) 
		{
			if (i == 0) 
			{
				this->layers[i].neurons[j].w = new double[npl[i]];
				this->layers[i].neurons[j].deltaW = new double[npl[i]];
				this->layers[i].neurons[j].wCopy = new double[npl[i]];
				this->layers[i].neurons[j].lastDeltaW = new double[npl[i]];
			}
			else 
			{
				this->layers[i].neurons[j].w = new double[npl[i - 1] + 1];
				this->layers[i].neurons[j].deltaW = new double[npl[i - 1] + 1];
				this->layers[i].neurons[j].wCopy = new double[npl[i - 1] + 1];
				this->layers[i].neurons[j].lastDeltaW = new double[npl[i - 1] + 1];
			}
		}
	}

	return 1;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	for(int i=0; i<this->nOfLayers;i++)
	{
		for(int j=0; j<this->layers[i].nOfNeurons; j++)
		{
			delete[] this->layers[i].neurons[j].w;
			delete[] this->layers[i].neurons[j].deltaW;
			delete[] this->layers[i].neurons[j].wCopy;
		}
		delete[] this->layers[i].neurons;
	}
	delete[] this->layers;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	for (int i = 1; i < this->nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i-1].nOfNeurons + 1; k++)
			{
				this->layers[i].neurons[j].w[k] = util::randomDouble(-1, 1);
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
	for (int i = 0; i < this->layers[0].nOfNeurons; i++)
	{
		this->layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
	for (int i = 0; i < this->layers[this->nOfLayers-1].nOfNeurons; i++)
	{
		output[i] = this->layers[this->nOfLayers-1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
	for (int i = 1; i < this->nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i-1].nOfNeurons + 1; k++)
			{
				this->layers[i].neurons[j].wCopy[k] = this->layers[i].neurons[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	for (int i = 1; i < this->nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i-1].nOfNeurons + 1; k++)
			{
				this->layers[i].neurons[j].w[k] = this->layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	double net;
	for(int i=1; i<this->nOfLayers; i++)
	{
		for(int j=0;j<this->layers[i].nOfNeurons; j++)
		{
			net = 0.0;
			for(int k=1; k<this->layers[i-1].nOfNeurons + 1; k++)
			{
				net += this->layers[i].neurons[j].w[k] * this->layers[i-1].neurons[k-1].out;
			}
			net += this->layers[i].neurons[j].w[0];
			this->layers[i].neurons[j].out = 1.0 / (1 + exp(-net));
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double* target) {
	double mse = 0.0;
	for(int i=0; i<this->layers[this->nOfLayers-1].nOfNeurons; i++)
	{
		mse += pow(target[i] - this->layers[this->nOfLayers - 1].neurons[i].out, 2);
	}
	mse /= this->layers[this->nOfLayers - 1].nOfNeurons;
	return mse;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double* target) {
	double out, aux;

	for(int i=0; i<this->layers[this->nOfLayers-1].nOfNeurons; i++)
	{
		out = this->layers[nOfLayers-1].neurons[i].out;
		this->layers[this->nOfLayers-1].neurons[i].delta = -(target[i]-out)*out*(1-out);
	}

	for(int i=this->nOfLayers-2; i>=1; i--)
	{
		for( int j=0; j<this->layers[i].nOfNeurons; j++)
		{
			out = this->layers[i].neurons[j].out;
			aux = 0.0;
			for(int k=0; k<this->layers[i+1].nOfNeurons; k++)
			{
				aux += this->layers[i+1].neurons[k].w[j+1] * this->layers[i+1].neurons[k].delta;
			}

			this->layers[i].neurons[j].delta = aux * out * (1 - out);
		}
	}
}


// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for(int i=1; i<this->nOfLayers; i++)
	{
		for(int j=0; j<this->layers[i].nOfNeurons; j++)
		{
			for(int k=1; k<this->layers[i-1].nOfNeurons +1; k++)
			{
				this->layers[i].neurons[j].deltaW[k] += this->layers[i].neurons[j].delta * this->layers[i-1].neurons[k-1].out;
			}
			this->layers[i].neurons[j].deltaW[0] += this->layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	for(int i=1; i<this->nOfLayers; i++)
	{
		for(int j=1; j<this->layers[i].nOfNeurons; j++)
		{
			for(int k=1; k<this->layers[i-1].nOfNeurons +1; k++)
			{
				this->layers[i].neurons[j].w[k] -= (this->eta*this->layers[i].neurons[j].deltaW[k])
				- (this->mu * this->eta * this->layers[i].neurons[j].lastDeltaW[k]); // Update the weight) ;
			}
			this->layers[i].neurons[j].w[0] -= (this->eta*this->layers[i].neurons[j].deltaW[0])
			- (this->mu * this->eta * this->layers[i].neurons[j].lastDeltaW[0]);
		}	
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	for(int i=1; i<this->nOfLayers; i++)
	{
		cout << "Layer " << i << endl;
		for(int j=0; j<this->layers[i].nOfNeurons; j++)
		{
			for(int k=0; k<this->layers[i].nOfNeurons + 1; k++)
			{
				cout << this->layers[i].neurons[j].w[k] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double* input, double* target) {
	this->feedInputs(input);
	this->forwardPropagate();
	this->backpropagateError(target);
	this->accumulateChange();
	this->weightAdjustment();
}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset* trainDataset) {
	int i;
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset* testDataset) {
	double mse = 0.0;
	double error;
	for(int i=0; i<testDataset->nOfPatterns; i++)
	{
		this->feedInputs(testDataset->inputs[i]);
		this->forwardPropagate();
		for(int j=0; j<testDataset->nOfOutputs; j++)
		{
			error = testDataset->outputs[i][j] - this->layers[this->nOfLayers-1].neurons[j].out;
			mse += error * error;
		}
	}
	mse /= testDataset->nOfPatterns;
	return mse;
}


// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset* pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * obtained = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nOfPatterns; i++){

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);
		
		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;

	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset * trainDataset, Dataset * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving;
	double testError = 0;

	// Learning
	do {

		trainOnline(trainDataset);
		double trainError = test(trainDataset);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}


		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nOfPatterns; i++){
		double* prediction = new double[pDatosTest->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<pDatosTest->nOfOutputs; j++)
			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	testError = test(pDatosTest);
	*errorTest=testError;
	*errorTrain=minTrainError;

}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
