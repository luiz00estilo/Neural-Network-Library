#pragma once
#ifndef NNLIB_H_
#define NNLIB_H_


long double ldRand(long double min, long double max);	//Returns a number int the range [min, max]

	/*Nodes have information about each node it is connected ("nodes") that is connected to, the "weights" of these connection (range [-1, 1]),
	how many nodes it is connected to ("outLen"), the value of which is was received from other nodes ("inputValue"), and its own "value" (range [-1, 1])*/
class node {
	friend class layer;
	friend class oldNeuralNetwork;
private:
	long double inputValue;
	long double value;
	long double* weights;
	node** outputs;
	int outLen;

public:
	//Creates node with "0" as "value"
	node();
	//Creates node with "d" as "value"
	node(long double d);
	~node();
	/*Copies the node stats
	-Note that it doesn't copies the quantity of connections, if it doesn't match, it'll copy what is possible (not recommended)*/
	void operator=(node& outNode);
	void invalidate();
	//Parses the "inputValue" in to a error function and outputs to "value" and sets "inputValue" to '0'
	void processInput();
	//Create a new connetion with a Weight
	void connect(node& n);
	//Transmits the value thought the weights to all connections
	void transmit();
	/*Mutates the weights value in the range [-maxMutationRate, maxMutationRate]
	-"maxMutationRate" is supposed to be in the range [-1, 1] (unexpected results otherwise)*/
	//void mutate(long double maxMutationRate);
	long double getValue();
	long double getWeight(int n);
};


class layer {
	friend class oldNeuralNetwork;
private:
	node** nodes;
	int nodesLen;
	layer* output;
public:
	layer(int len);
	layer(int len, long double* inputs);
	~layer();
	/*Copies the nodes stats
	-Note that it doesn't copies the lenght, if it doesn't match, it'll copy what is possible (not recommended)*/
	void operator=(layer &outLayer);
	//Returns a pointer to node "n" on the layer
	node* operator[](int n);
	void invalidate();
	void processInputs();
	//Connects all the nodes in "this" layer to the nodes in "l" layer
	void connect(layer& l);
	void transmit();
	//void mutate(long double maxMutationRate);
	int getLen();
	//Returns a pointer to node "n" on the layer
	node* getNode(int n);

};



class oldNeuralNetwork {
private:
	layer** layers/* = new layer**/;
	int layersLen;
	long double mutationRate;

public:
	/*-The "Len" variables determine how many nodes will be in that type of layer (all the hidden layers have the same amount of nodes)
	-The Default Maximum Mutation Ratio can be changed in the header file*/
	oldNeuralNetwork(int inputLen, int hiddenLen, int outputLen, int numberOfHiddenLayers, long double limitMaxMutationRate = 0.05);
	//Creates a copy of the input network
	oldNeuralNetwork(oldNeuralNetwork& net);
	~oldNeuralNetwork();
	//Will return a pointer to layer "l" on the network (from {input -> hidden -> output} order)
	layer* operator[](int l);
	void operator=(oldNeuralNetwork& net);
	void invalidate();
	//Returns a pointer to a new network, which is a mutation of the two networks involved
	oldNeuralNetwork* operator+(oldNeuralNetwork &net);
	//Inputs the "inputs" into the nodes in the input layer
	void input(long double* inputs);
	void output(long double * outputs);
	//Transmits the input thought the newtork ending at the output
	void process();
	/*Turns "net" into a mutated version of "this" network
	-This is, usually, the faster version of the function*/
	void getMutation(oldNeuralNetwork& net);
	/*Returns a mutated version of "this" network
	-This is, usually, the slower version of the function*/
	oldNeuralNetwork getMutation();
	//Will return a pointer to layer "l" on the network (from {input -> hidden -> output} order)
	layer* getLayer(int l);
	//Will show all Values and Weights
	void show();

};


class neuralNetwork {
	long double** value;
	long double*** weight;
	int netLen;
	int* layerLen;
public:
	void show();
};



#endif
