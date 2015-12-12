#include "util.h"
#include "Parser.h"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>

using namespace std;

int main(int argc, char** argv){

	
	if( argc < 3 ){
		cerr << "Usage: svrPredict [testData] [modelFile]\n";
		exit(0);
	}
	
	char* trainFile = argv[1];
	char* modelFile = argv[2];
	
	//read model
	ifstream fin(modelFile);
	int D, nnz, ind;
	double val;
	fin >> D >> nnz;
	double* w = new double[D];
	for(int i=0;i<D;i++)
		w[i] = 0.0;
	for(int i=0;i<nnz;i++){
		fin >> ind >> val;
		w[ind] = val;
	}
	//read data
	vector<SparseVec*> X;
	vector<double> y;
	Parser::parseSVM(trainFile,D, X, y);
	int N = X.size();

	//predict
	double* prediction = new double[N];
	for(int i=0;i<N;i++){
		prediction[i] = dot(w, X[i]);
	}
	
	//report RMSE and output prediction
	double Acc=0.0;
	for(int i=0;i<N;i++){
		double ywx = y[i] * prediction[i];
		if( ywx > 0.0 )
			Acc += 1.0;
	}
	Acc = Acc/N;

	cerr << "Acc=" << Acc << endl;
}
