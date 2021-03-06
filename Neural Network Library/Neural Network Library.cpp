// NeuralNetwork.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <iostream>
#include <random>
#include <chrono>
#include "NNLib.h"
using std::cout;
using std::endl;

int iRand(int min, int max) {
	std::uniform_int_distribution<int> rnd(min, max);
	std::default_random_engine eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	return rnd(eng);
}

void quickSort(oldNeuralNetwork** mainNets, long double* missRates, oldNeuralNetwork* tempNet, int left, int right) {
	int l = left;
	int r = right;
	long double temp;
	long double mid = missRates[(left + right) / 2];

	while (l <= r) {
		while (missRates[l] < mid) l++;
		while (missRates[r] > mid) r--;
		if (l <= r) {
			temp = missRates[l];
			missRates[l] = missRates[r];
			missRates[r] = temp;
			tempNet[0] = mainNets[l][0];
			mainNets[l][0] = mainNets[r][0];
			mainNets[r][0] = tempNet[0];
			l++;
			r--;
		}
	}

	if (left < r) quickSort(mainNets, missRates, tempNet, left, r);
	if (l < right) quickSort(mainNets, missRates, tempNet, l, right);

}

void sortNets(oldNeuralNetwork** mainNets, long double* missRates, int lenght) {	//sorts the array
	//bool done = false;
	oldNeuralNetwork tempNet(mainNets[0][0]);

	quickSort(mainNets, missRates, &tempNet, 0, (lenght - 1));
	//while (done == false) {
	//	done = true;
	//	for (int x = 0; x < (lenght - 1); x++) {
	//		if (missRates[x] > missRates[x + 1]) {
	//			long double i = missRates[x + 1];	//}
	//			missRates[x + 1] = missRates[x];	//}Swaps the scores
	//			missRates[x] = i;					//}
	//			tempNet = mainNets[x + 1][0];			//}
	//			mainNets[x + 1][0] = mainNets[x][0];//}Swaps the networks
	//			mainNets[x][0] = tempNet;				//}
	//			done = false;
	//		}
	//	}
	//}
}
long double getCost(long double target, long double result) {
	long double miss = result - target;
	/*if (miss < 0)*/ miss = miss * miss;
	return miss;
}
class nnPtr {

};

int main()
{
	//neuralNetwork n1(2, 2, 2, 2, 0.001);
	//neuralNetwork n2(2, 2, 2, 2, 0.001);
	//neuralNetwork* n3;
	//n1.show();
	//n2.show();
	////n3->show();
	//while (true) {
	//	//for (int x = 0; x < 1000; x++) {
	//		n3 = n1 + n2;
	//		//cout << x << endl;
	//	//}
	//	cout << "Done\n";
	//	n1.show();
	//	n2.show();
	//	n3->show();
	//	std::cin.ignore();
	//}
	/*long double n;
	while (true) {
		for (int x = 0; x < 100; x++) {
			n = iRand(0, 1);
			cout << n << endl;
		}
		std::cin.ignore(1000, '\n');
	}*/
	/*long double l[10] = { 1.5,0.8,3,7,6,9,2,0,4,10 };
	neuralNetwork** n = new neuralNetwork*[10];
	for (int x = 0; x < 10; x++) n[x] = new neuralNetwork(1, 1, 1, 1);
	for (int x = 0; x < 10; x++) {
		cout << l[x] << endl;
		n[x]->show();
		cout << endl;
	}
	cout << endl << endl;
	sortNets(n, l, 10);
	for (int x = 0; x < 10; x++) {
		cout << l[x] << endl;
		n[x]->show();
		cout << endl;
	}
	std::cin.ignore(1000, '\n');*/
	/*neuralNetwork n0(2, 2, 2, 2, 0.1);
	neuralNetwork n1(1, 1, 1, 1);;
	n0.show();
	n1.show();
	n1 = n0.getMutation();
	n0.show();
	n1.show();*/

	/*
	std::uniform_int_distribution<int> rnd(0, 1);															//}Creating random generator
	std::default_random_engine eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());	//}for test creation

	int testLen = 20;
	int netsLen = 1000;
	int inLen = 3 + 1;
	int outLen = 1;
	int passNets = ((netsLen * 10) / 100);
	long double target;
	long double* input = new long double[inLen];
	long double* output = new long double[outLen];
	long double* costRates = new long double[netsLen];
	oldNeuralNetwork** nets = new oldNeuralNetwork*[netsLen];
	for (int x = 0; x < netsLen; x++) nets[x] = new oldNeuralNetwork(inLen, 12, outLen, 2, 0.8);
	input[inLen - 1] = 1;

start:
	for (int genCounter = 0; genCounter < 20; genCounter++) {
		cout << genCounter << ":Testing...";
		for (int x = 0; x < netsLen; x++) costRates[x] = 0;
		for (int x = 0; x < testLen; x++) {	//For the lenght of the test
			input[0] = rnd(eng);											//}
			input[1] = rnd(eng);											//}
			input[2] = rnd(eng);											//}Test Design
			if (input[0] == input[1] && input[1] == input[2]) target = 1;	//}
			else target = 0;												//}

			for (int y = 0; y < netsLen; y++) {	//For all nets
				nets[y]->input(input);		//}
				nets[y]->process();			//}Testing all nets
				nets[y]->output(output);	//}
				costRates[y] += getCost(target, output[0]);	//Gets results
				/*cout << input[0] << input[1] << ' ' << output[0] << ' ' << target << ' ' << missRates[y] << endl;
				nets[x]->show();
				std::cin.ignore(1000, '\n');*//*/
			}
		}
		for (int x = 0; x < netsLen; x++) costRates[x] = costRates[x] / testLen;	//Getting the medium of the results
		/*for (int x = 0; x < netsLen; x++) {
			long double misses = 0;
			for (int y = 0; y < 40; y++) {
				input[0] = iRand(0, 1);
				input[1] = iRand(0, 1);
				if (input[0] == input[1]) target = 1;
				else target = 0;
				nets[x]->input(input);
				nets[x]->process();
				nets[x]->output(output);
				misses += getMiss(target, output[0]);
			}
			missRates[x] = misses / 10;

		}*//*
		cout << "Sorting.../";
		sortNets(nets, costRates, netsLen);	//Sorting the nets
		/*for (int x = 0; x < netsLen; x++) cout << missRates[x] << endl;
		std::cin.ignore(1000, '\n');*//*
		cout << "Mutating..." << endl;

		for (int x = passNets; x < netsLen;) {							//}
			for (int y = 0; y < passNets || x < netsLen; y++, x++) {	//}
				nets[y][0].getMutation(nets[x][0]);						//}Mutating the best nets 
			}															//}defined by passNets
		}																//}
	}


	//for (int x = 0; x < netsLen; x++) cout << missRates[x] << endl;
	int done = 1;
	int test = 0;
	for (int x = 0; x < 100; x++) {
		input[0] = rnd(eng);											//}
		input[1] = rnd(eng);											//}
		input[2] = rnd(eng);											//}Test Design
		if (input[0] == input[1] && input[1] == input[2]) target = 1;	//}
		else target = 0;												//}
		nets[0]->input(input);		//}
		nets[0]->process();			//}Processing inputs
		nets[0]->output(output);	//}
		if (output[0] > 0.95) test = 1;								//}
		else if (output[0] > -0.05 && 0.05 > output[0]) test = 0;	//}Accuracy check design
		else test = 2;												//}
		cout << input[0] << input[1] << input[2] << ' ' << target << ' ' << output[0] << ' ';
		if (test == target) cout << "correct" << endl;
		else {
			cout << "wrong" << endl;
			done = 0;
		}
	}
	*/
	
	/*int*** temp;
	for (int a = 0; true; a++) {
		temp = new int**[10];
		for (int x = 0; x < 10; x++) {
			temp[x] = new int*[20];
			for (int y = 0; y < 20; y++) temp[x][y] = new int[30];
		}

		for (int x = 0; x < 10; x++) {
			for (int y = 0; y < 20; y++) delete[] temp[x][y];
			delete[] temp[x];
		}
		delete[] temp;
		cout << a << ' ';
	}*/
	
	//neuralNetwork** net;
	//net = new neuralNetwork*[1000];
	//int layersLen[4] = {3, 2, 2, 3};
	long double inputs[10] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 };
	long double outputs = 0;
	for (int x = 0; true; x++) {
		try {
			neuralNetwork net(5, 10, 4, 2);
			net.input(inputs);
			net.feedForward();
			net.show();
			//cout << outputs << endl;
			std::cin.ignore(1000, '\n');
		}
		catch (const std::exception& expt) {
			cout << "Exeption found:" << expt.what() << ". Initialization Cancelled" << endl;
		}
		cout << x << ' ';
	}

	/*if (done == 1)*/ std::cin.ignore(1000, '\n');
	// goto start;
	return 0;
}

