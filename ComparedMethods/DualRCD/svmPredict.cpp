#include "Parser.h"
#include "Instance.h"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>

using namespace std;

double dot( double* w, vector<pair<int,double> >& x ){
	
	double sum=0.0;
	for(int i=0;i<x.size();i++){
	
		int index = x[i].first;
		double value = x[i].second;

		sum += w[index]*value;
	}
	
	return sum;
}


int main(int argc, char** argv){

	
	if( argc < 3 ){
		cerr << "Usage: svrPredict [testData] [modelFile]\n";
		exit(0);
	}
	
	char* trainFile = argv[1];
	char* modelFile = argv[2];
	
	//read model
	ifstream fin(modelFile);
	int D, nnz;
	fin >> D >> nnz;
	double* w = new double[D];
	for(int i=0;i<D;i++){
		w[i] = 0.0;
	}
	int ind;
	double val;
	for(int i=0;i<nnz;i++){
		fin >> ind >> val;
		w[ind] = val;
	}
	//read data
	vector<Instance*>* data =  Parser::parseSVM(trainFile,D);
	int N = data->size();

	//predict
	double* prediction = new double[N];
	for(int i=0;i<N;i++){
		prediction[i] = dot(w,data->at(i)->xi);
	}
	
	//report RMSE and output prediction
	double Acc=0.0;
	for(int i=0;i<N;i++){
		double ywx = data->at(i)->yi* prediction[i];
		if( ywx > 0.0 )
			Acc += 1.0;
	}
	Acc = Acc/N;

	cerr << "Acc=" << Acc << endl;
}
