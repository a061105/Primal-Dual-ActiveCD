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
double prox_l1( double v, double lambda ){
	
	if( fabs(v) > lambda ){
		if( v>0.0 )
			return v - lambda;
		else 
			return v + lambda;
	}
	
	return 0.0;
}

double objective(double* w, int D, double* alpha, int N){
	
	double obj = 0.0;
	for(int i=0;i<D;i++)
		obj += w[i]*w[i];
	obj /= 2.0;

	for(int i=0;i<N;i++)
		obj -= alpha[i];

	for(int i=0;i<N;i++){
		obj += alpha[i]*alpha[i]/2.0;
	}

	return obj;
}

int nnz(double* v, int size){
	
	int count=0;
	for(int i=0;i<size;i++)
		if( fabs(v[i]) > 1e-12 )
			count++;
	return count;
}

int main(int argc, char** argv){

	
	if( argc < 1+2 ){
		cerr << "Usage: svmTrain [train file] [lambda] (modelFile)\n";
		exit(0);
	}
	
	char* trainFile = argv[1];
	double lambda = atof(argv[2]);
	char* modelFile;
	if( argc > 1+2 )
		modelFile = argv[3];
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
	double* w = new double[D];
	double* alpha = new double[N];
	
	for(int i=0;i<D;i++)
		v[i] = 0;
	for(int i=0;i<D;i++)
		w[i] = 0;
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

	int max_iter = 10000;
	int iter=0;
	while(iter < max_iter){
		
		double update_time = -omp_get_wtime();
		for(int r=0;r<N;r++){
			
			int i = index[r];
			
			double yi = data->at(i)->yi;
			//1. compute gradient
			double gi = yi*dot(w,data->at(i)->xi) - 1.0 + alpha[i];
			//2. compute new alpha
			double new_alpha = min( max( alpha[i] - gi/Qii[i] , 0.0 ) , C);
			//3. maintain w and v
			double alpha_diff = new_alpha-alpha[i];
			if(  fabs(alpha_diff) > 1e-12 ){
				
				Instance* ins = data->at(i);
				int ind;
				double value;
				for(int k=0;k<ins->xi.size();k++){
					
					ind = ins->xi[k].first;
					value = ins->xi[k].second;
					
					v[ind] += alpha_diff * (yi*value);
					w[ind] = prox_l1( v[ind], lambda );
				}
				
				alpha[i] = new_alpha;
			}
		}
		update_time += omp_get_wtime();

		//if(iter%10==0)
		cerr << "iter=" << iter << ", nnz_a=" << nnz(alpha,N) << ", nnz_a=" << nnz(w,D) << ", obj=" << objective(w,D,alpha,N) << ", time=" << update_time << endl ;
		
		shuffle(index);
		iter++;
	}
	cerr << endl;
	
	int nnz=0;
	for(int j=0;j<D;j++){
		if( fabs(w[j]) > 1e-12 )
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
		if( fabs(w[i]) > 1e-12 )
			fout << i << " " << w[i] << endl;
	fout.close();
}
