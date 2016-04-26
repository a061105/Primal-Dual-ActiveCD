#include "Parser.h"
#include "Instance.h"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>

using namespace std;

template<class T>
void shuffle( vector<T>& vect ){
	
	int r;
	for(int i=0;i<vect.size()-1;i++){
		r =  (rand() % (vect.size() - i-1))+1+i ;
		swap(vect[i],vect[r]);
	}
}

double dot( double* v, vector<pair<int,double> >& x ){
	
	double sum=0.0;
	for(int i=0;i<x.size();i++){
	
		int index = x[i].first;
		double value = x[i].second;

		sum += v[index]*value;
	}
	
	return sum;
}


int main(int argc, char** argv){

	
	if( argc < 2 ){
		cerr << "Usage: svmTrain [train file] (modelFile)\n";
		exit(0);
	}
	
	char* trainFile = argv[1];
	char* modelFile;
	if( argc >= 3 )
		modelFile = argv[2];
	else{
		modelFile = "model";
	}
	
	double C = 1.0;
	int D;
	int N;
	vector<Instance*>* data =  Parser::parseSVM(trainFile,D);
	N = data->size();
	cerr << "N=" << N << endl;
	cerr << "D=" << D << endl;
	//initialization
	double* v = new double[D];
	double* alpha = new double[N];
	
	for(int i=0;i<D;i++)
		v[i] = 0;
	for(int i=0;i<N;i++){
		alpha[i] = 0;
	}
	
	//Compute diagonal of Q matrix
	double* Qii = new double[N];
	for(int i=0;i<N;i++){
		
		Instance* ins = data->at(i);
		
		Qii[i] = 0;
		for(int j=0;j<ins->xi.size();j++){
			double value = ins->xi[j].second;
			Qii[i] += value*value;
		}
	}
	
	//Main Loop
	vector<int> index;
	for(int i=0;i<N;i++)
		index.push_back(i);
	shuffle(index);

	int max_iter = 300;
	int iter=0;
	while(iter < max_iter){
		
		double update_time = -omp_get_wtime();
		for(int r=0;r<N;r++){
			
			int i = index[r];
			double yi = data->at(i)->yi;
			//1. compute gradient of i 
			double gi = yi*dot(v,data->at(i)->xi) - 1.0;
			//2. compute alpha_new
			double new_alpha = min( max( alpha[i] - gi/Qii[i] , 0.0 ) , C);
			//3. maintain v (=w)
			double alpha_diff = new_alpha-alpha[i];
			if(  fabs(alpha_diff) > 1e-5 ){
				
				Instance* ins = data->at(i);
				int index;
				double value;
				for(int k=0;k<ins->xi.size();k++){
					
					index = ins->xi[k].first;
					value = ins->xi[k].second;
					
					v[index] += alpha_diff * (yi*value);
				}
				
				alpha[i] = new_alpha;
			}
		}
		update_time += omp_get_wtime();

		//if(iter%10==0)
		cerr << "iter=" << iter << ", time=" << update_time << endl ;
		
		shuffle(index);
		iter++;
	}
	cerr << endl;
	
	int nnz=0;
	for(int j=0;j<D;j++){
		if( fabs(v[j]) > 1e-12 )
			nnz++;
	}
	cerr << "w_nnz=" << nnz << endl;
	int SV=0;
	for(int i=0;i<N;i++){
		if( fabs(alpha[i]) > 1e-12 )
			SV++;
	}

	cerr << "alpha_nnz=" << SV << endl;

	//output model
	ofstream fout(modelFile);
	fout << D << " " << nnz << endl;
	for(int i=0;i<D;i++)
		if( fabs(v[i]) > 1e-12 )
			fout << i << " " << v[i] << endl;
	fout.close();
}
