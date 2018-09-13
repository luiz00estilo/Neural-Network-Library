#include "NNLib.h"
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>


long double ldRand(long double min, long double max) {
	std::uniform_real_distribution<long double> rnd(min, max);
	std::default_random_engine eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	return rnd(eng);
}


node::node() {
	inputValue = 0;
	value = 0;
	weights = new long double;
	outputs = new node*;
	outLen = 0;
}
node::node(long double d)/*creates node with "d" as "value"*/ {
	inputValue = 0;
	value = d;
	weights = new long double;
	outputs = new node*;
	outLen = 0;
}
node::~node() {
	delete[] outputs;
	delete[] weights;
}
void node::operator=(node& outNode) {
	this->inputValue = outNode.inputValue;
	this->value = outNode.value;
	for (int x = 0; x < this->outLen && x < outNode.outLen; x++) this->weights[x] = outNode.weights[x];
}
void node::invalidate() {
	inputValue = 0;
	value = 0;
	for (int x = 0; x < outLen; x++) weights[x] = 0;
}
void node::processInput() /*parses the "inputValue" in to a error function and outputs to "value"*/ {
	value = erf(inputValue);
	inputValue = 0;
}
void node::connect(node& n) {
	node** newOutputs = new node*[outLen + 1];				//|Creating the new set of nodes, which will include the new node
	long double* newWeights = new long double[outLen + 1];	//|                        weights                           weight
	for (int x = 0; x < outLen; x++) {	//|
		newOutputs[x] = outputs[x];		//|
		newWeights[x] = weights[x];		//|Copying the present connections
	}									//|
	newOutputs[outLen] = &n;			//Adding new node
	newWeights[outLen] = ldRand(-1, 1);	//Generating new weight (range [-1,1])
	delete[] outputs;		//|
	delete[] weights;		//|Redirecting pointers
	weights = newWeights;	//|
	outputs = newOutputs;	//|
	outLen++;
}
void node::transmit() {
	for (int x = 0; x < outLen; x++) {
		outputs[x]->inputValue += (value * weights[x]);
	}
}
//void node::mutate(long double mutationRate)  {
//	for (int x = 0; x < outLen; x++) {
//		long double newWeight = weights[x] + ldRand(-mutationRate, mutationRate);
//		if (-1 < newWeight && newWeight < 1) weights[x] = newWeight;
//	}
//}
long double node::getValue() {
	return value;
}
long double node::getWeight(int n) {
	if (n >= 0 && n < outLen) {
		return weights[n];
	}
	return -40004;
}




layer::layer(int len) {
	if (len < 1) len = 1;
	nodes = new node*[len];
	for (int x = 0; x < len; x++) nodes[x] = new node;
	nodesLen = len;
}
layer::layer(int len, long double* inputs) {
	if (len < 1) len = 1;
	nodes = new node*[len];
	for (int x = 0; x < len; x++) nodes[x] = new node(inputs[x]);
	nodesLen = len;
}
layer::~layer() {
	for (int x = 0; x < nodesLen; x++) delete nodes[x];
	delete[] nodes;
}
void layer::operator=(layer &outLayer) {
	for (int x = 0; x < this->nodesLen && x < outLayer.nodesLen; x++) this->nodes[x][0] = outLayer[x][0];
}
node* layer::operator[](int n) {
	if (n >= 0 && n < nodesLen) return nodes[n];
	else return NULL;
}
void layer::invalidate() {
	for (int x = 0; x < this->nodesLen; x++) this->nodes[x]->invalidate();
}
void layer::processInputs() {
	for (int x = 0; x < nodesLen; x++) nodes[x]->processInput();
}
void layer::connect(layer& l) {
	output = &l;
	for (int x = 0; x < nodesLen; x++) {
		for (int y = 0; y < this->output->nodesLen; y++) this->nodes[x]->connect(output->nodes[y][0]);
	}
}
void layer::transmit() {
	for (int x = 0; x < nodesLen; x++) nodes[x]->transmit();
}
//void layer::mutate(long double maxMutationRate) {
//	for (int x = 0; x < nodesLen; x++) nodes[x]->mutate(maxMutationRate);
//}
int layer::getLen() {
	return nodesLen;
}
node* layer::getNode(int n) {
	if (n >= 0 && n < nodesLen) return nodes[n];
	else return NULL;
}




oldNeuralNetwork::oldNeuralNetwork(int inputLen, int hiddenLen, int outputLen, int numberOfHiddenLayers, long double newMutationRate) {
	if (inputLen < 1) inputLen = 1;							//}
	if (hiddenLen < 1) hiddenLen = 1;						//}
	if (outputLen < 1) outputLen = 1;						//} Checkings
	if (numberOfHiddenLayers < 1) numberOfHiddenLayers = 1;	//}
	layers = new layer*[2 + numberOfHiddenLayers];	//Creating new layer pointers
	layers[0] = new layer(inputLen);														//Generating input layer
	for (int x = 0; x < numberOfHiddenLayers; x++) layers[1 + x] = new layer(hiddenLen);	//Generating hidden layers
	layers[1 + numberOfHiddenLayers] = new layer(outputLen);								//Generating output layer
	layersLen = 2 + numberOfHiddenLayers;	//Sets the network lenght
	for (int x = 1; x < layersLen; x++) layers[x - 1]->connect(layers[x][0]);	//Connects all layers
	mutationRate = newMutationRate;	//Sets the mutationRate
}
oldNeuralNetwork::oldNeuralNetwork(oldNeuralNetwork& net) {
	layers = new layer*[net.layersLen];	//Generatins new layer pointers
	layers[0] = new layer(net.layers[0]->nodesLen);														//|Generating input layers
	for (int x = 0; x < (net.layersLen - 2); x++) layers[1 + x] = new layer(net.layers[1]->nodesLen);	//|           hidden
	layers[1 + (net.layersLen - 2)] = new layer(net.layers[net.layersLen - 1]->nodesLen);				//|           output
	for (int x = 1; x < net.layersLen; x++) layers[x - 1]->connect(layers[x][0]);	//Connects all layers
	for (int x = 0; x < net.layersLen; x++) this->layers[x][0] = net.layers[x][0];	//}
	this->layersLen = net.layersLen;												//}Copies the stats from "net" to "this"
	this->mutationRate = net.mutationRate;											//}
}
oldNeuralNetwork::~oldNeuralNetwork() {
	for (int x = 0; x < layersLen; x++) delete layers[x];
	delete[] layers;
}
layer* oldNeuralNetwork::operator[](int l) {
	if (l >= 0 && l < layersLen) return layers[l];
	else return NULL;
}
void oldNeuralNetwork::operator=(oldNeuralNetwork& net) {
	if (this->layersLen != net.layersLen ||															//testing if the number of layers is equal
		this->layers[0]->nodesLen != net.layers[0]->nodesLen ||											//|testing if the number of nodes in the input layer are equal
		this->layers[1]->nodesLen != net.layers[1]->nodesLen ||											//|                                      hidden
		this->layers[(this->layersLen - 1)]->nodesLen != net.layers[(net.layersLen - 1)]->nodesLen) {	//|                                      output
		/*for (int x = 0; x < this->layersLen && x < outNet.layersLen; x++) this->layers[x][0] = outNet.layers[x][0];
		this->mutationRate = outNet.mutationRate;*/

		for (int x = 0; x < this->layersLen; x++) delete this->layers[x];	//}Deleting old layers
		delete[] this->layers;												//}

		this->layers = new layer*[net.layersLen];	//Creating new layer pointers
		this->layers[0] = new layer(net.layers[0]->nodesLen);												//Generating input layer
		for (int x = 1; x < (net.layersLen - 1); x++) this->layers[x] = new layer(net.layers[1]->nodesLen);	//Generating hidden layers
		this->layers[net.layersLen - 1] = new layer(net.layers[net.layersLen - 1]->nodesLen);				//Generating output layer
		for (int x = 1; x < net.layersLen; x++) this->layers[x - 1]->connect(this->layers[x][0]);	//Connects all layers
	}
	for (int x = 0; x < net.layersLen; x++) this->layers[x][0] = net.layers[x][0];	//}
	this->layersLen = net.layersLen;												//}Copies the stats from "net" to "this"
	this->mutationRate = net.mutationRate;											//}
}
oldNeuralNetwork* oldNeuralNetwork::operator+(oldNeuralNetwork &net) {
	if (this->mutationRate == 0 || net.mutationRate == 0 ||															//testing if either of the nets are infertile
		this->layersLen != net.layersLen ||																			//testing if the number of layers is equal
		this->layers[0]->nodesLen != net.layers[0]->nodesLen ||														//|testing if the number of nodes in the input layer are equal
		this->layers[1]->nodesLen != net.layers[1]->nodesLen ||														//|                                      hidden
		this->layers[(this->layersLen - 1)]->nodesLen != net.layers[(net.layersLen - 1)]->nodesLen) return nullptr;	//|                                      output

	oldNeuralNetwork* resultNet = new oldNeuralNetwork(this->layers[0]->nodesLen, this->layers[1]->nodesLen, this->layers[(this->layersLen - 1)]->nodesLen, (this->layersLen - 2));	//Creates a new network with the same size as "this" network

	for (int x = 0; x < (this->layersLen - 1); x++) {						//Loops the weight making thought all layers (exept the last)
		for (int y = 0; y < this->layers[x]->nodesLen; y++) {				//Loops thought all the nodes of the layer
			for (int z = 0; z < this->layers[x]->nodes[y]->outLen; z++) {	//Loops thought all the weights of the node
				do {
					resultNet->layers[x]->nodes[y]->weights[z] = ((this->layers[x]->nodes[y]->weights[z] + ldRand(-(this->mutationRate), this->mutationRate)) +	//}Sets the new weight of resultNet as the addition of: The mutation of "this"'s (parent 1) weight
						(net.layers[x]->nodes[y]->weights[z] + ldRand(-(net.mutationRate), net.mutationRate))) / 2;												//}and the mutation of the "net"'s weight divided by 2
				} while (resultNet->layers[x]->nodes[y]->weights[z] < -1 || resultNet->layers[x]->nodes[y]->weights[z] > 1);	//if the result is off the limits of [-1,1], it tries again
			}
		}
	}
	resultNet->mutationRate = (this->mutationRate + net.mutationRate) / 2;
	return resultNet;
}
void oldNeuralNetwork::invalidate() {
	for (int x = 0; x < this->layersLen; x++) this->layers[x]->invalidate();
	this->mutationRate = 0;
}
void oldNeuralNetwork::input(long double* inputs) {
	for (int x = 0; x < layers[0]->getLen(); x++) layers[0]->getNode(x)->value = inputs[x];
}
void oldNeuralNetwork::output(long double* outputs) {
	for (int x = 0; x < layers[layersLen - 1]->nodesLen; x++) outputs[x] = layers[layersLen - 1]->nodes[x]->value;
}
void oldNeuralNetwork::process() {
	for (int x = 0; x < (layersLen - 1); x++) {
		layers[x]->transmit();
		layers[x + 1]->processInputs();
	}
}
void oldNeuralNetwork::getMutation(oldNeuralNetwork& net) {
	//neuralNetwork resultNet(this->layers[0]->nodesLen, this->layers[1]->nodesLen, this->layers[(this->layersLen - 1)]->nodesLen, (this->layersLen - 2), this->mutationRate);	//Creates a new network with the same size as "this" network
	net = this[0];
	std::uniform_real_distribution<long double> rnd(-this->mutationRate, this->mutationRate);				//}Creating the random system
	std::default_random_engine eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());	//}
	for (int x = 0; x < (this->layersLen - 1); x++) {						//|Loops thought all the layers of the net (exept the output layer)
		for (int y = 0; y < this->layers[x]->nodesLen; y++) {				//|                      nodes         layer.
			for (int z = 0; z < this->layers[x]->nodes[y]->outLen; z++) {	//|                      weights       node.
				do {
					net.layers[x]->nodes[y]->weights[z] = this->layers[x]->nodes[y]->weights[z] + rnd(eng);	//Sets the new weight as a mutated version of "this"'s node
				} while (net.layers[x]->nodes[y]->weights[z] < -1 || net.layers[x]->nodes[y]->weights[z] > 1);	//if the result is off the limits of [-1,1], it tries again
			}
		}
	}
}
oldNeuralNetwork oldNeuralNetwork::getMutation() {
	oldNeuralNetwork resultNet(this->layers[0]->nodesLen, this->layers[1]->nodesLen, this->layers[(this->layersLen - 1)]->nodesLen, (this->layersLen - 2), this->mutationRate);	//Creates a new network with the same size as "this" network
	for (int x = 0; x < (this->layersLen - 1); x++) {						//Loops the weight making thought all layers (exept the last)
		for (int y = 0; y < this->layers[x]->nodesLen; y++) {				//Loops thought all the nodes of the layer
			for (int z = 0; z < this->layers[x]->nodes[y]->outLen; z++) {	//Loops thought all the weights of the node
				do {
					resultNet.layers[x]->nodes[y]->weights[z] = this->layers[x]->nodes[y]->weights[z] + ldRand(-this->mutationRate, this->mutationRate);	//Sets the new weight as mutates version the "this" one
				} while (resultNet.layers[x]->nodes[y]->weights[z] < -1 || resultNet.layers[x]->nodes[y]->weights[z] > 1);	//if the result is off the limits of [-1,1], it tries again
			}
		}
	}
	return resultNet;
}
layer* oldNeuralNetwork::getLayer(int l) {
	if (l >= 0 && l < layersLen) return layers[l];
	else return nullptr;
}
//neuralNetwork* neuralNetwork::reproduce(neuralNetwork &net) {
//	if (this->mutationRate == 0 || net.mutationRate == 0 ||															//testing if either of the nets are infertile
//	this->layersLen != net.layersLen ||																			//testing if the number of layers is equal
//	this->layers[0]->nodesLen != net.layers[0]->nodesLen ||														//|testing if the number of nodes in the input layer are equal
//	this->layers[1]->nodesLen != net.layers[1]->nodesLen ||														//|                                      hidden
//	this->layers[(this->layersLen - 1)]->nodesLen != net.layers[(net.layersLen - 1)]->nodesLen) return nullptr;	//|                                      output
//	
//	neuralNetwork* tempNet = new neuralNetwork(this->layers[0]->nodesLen, this->layers[1]->nodesLen, this->layers[(this->layersLen - 1)]->nodesLen, (this->layersLen - 2));	//Creates a new network with the same size as "this" network
//
//	for (int x = 0; x < (this->layersLen - 1); x++) {						//Loops the weight making thought all layers (exept the last)
//		for (int y = 0; y < this->layers[x]->nodesLen; y++) {				//Loops thought all the nodes of the layer
//			for (int z = 0; z < this->layers[x]->nodes[y]->outLen; z++) {	//Loops thought all the weights of the node
//				do {
//					tempNet->layers[x]->nodes[y]->weights[z] = ((this->layers[x]->nodes[y]->weights[z] + ldRand(-(this->mutationRate), this->mutationRate)) +	//sets the new weight as the result of de addition of the mutation of "this" weight
//						(net.layers[x]->nodes[y]->weights[z] + ldRand(-(net.mutationRate), net.mutationRate))) / 2;	//plus the mutation of the "net" weight divided by 2
//
//					//std::cout << y << ' ' << tempNet->layers[0]->nodes[x]->weights[y] << std::endl;;
//				} while (tempNet->layers[x]->nodes[y]->weights[z] < -1 || tempNet->layers[x]->nodes[y]->weights[z] > 1);	//if the result is off the limits of [-1,1], it tries again
//			}
//		}
//	}
//	return tempNet;
//}
void oldNeuralNetwork::show() {
	if (this == nullptr) {		//Catching a null pointer
		std::cout << "Exeption: NN Null Pointer" << std::endl;
		return;
	}
	std::cout << "MMR:" << this->mutationRate << std::endl;
	for (int x = 0; x < layersLen; x++) {
		std::cout << 'V' << x << ':';
		for (int y = 0; y < layers[x]->nodesLen; y++) std::cout << layers[x]->nodes[y]->value << ' ';
		std::cout << std::endl << 'W' << x << ':';
		for (int y = 0; y < layers[x]->nodesLen; y++)
			for (int z = 0; z < layers[x]->nodes[y]->outLen; z++) std::cout << layers[x]->nodes[y]->weights[z] << ' ';
		std::cout << std::endl;
	}
}



