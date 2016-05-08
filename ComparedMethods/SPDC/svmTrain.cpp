#include "Parser.h"
#include "Instance.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
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

// double dot( double* v1, double* v2, int n ){
	
// 	double sum=0.0;
// 	for(int i = 0; i < n; ++i){
// 		sum += v1[i] * v2[i];
// 	}
// 	return sum;
// }

double prox_l1( double v, double lambda ){
	
	if( fabs(v) > lambda ){
		if( v>0.0 )
			return v - lambda;
		else 
			return v + lambda;
	}
	
	return 0.0;
}

// double objective(double* w, int D, double* alpha, int N){
	
// 	double obj = 0.0;
// 	for(int i=0;i<D;i++)
// 		obj += w[i]*w[i];
// 	obj /= 2.0;
	
// 	cout << "primal: " << obj << endl;
// 	double tmp = obj;
	
// 	for(int i=0;i<N;i++)
// 		obj -= alpha[i];

// 	cout << "second: " << obj-tmp << endl;
// 	tmp = obj;
	
// 	for(int i=0;i<N;i++){
// 		obj += alpha[i]*alpha[i]/2.0;
// 	}

// 	cout << "third: " << obj-tmp << endl;
// 	return obj;
// }

double smooth_hinge_loss(double z, double mu) {
	if (z > mu) {
		return 0;
	} else if (z < 1.0 - mu) {
		return 1.0 - z - 0.5*mu;
	} 
	return 0.5 / mu * (1.0 - z) * (1.0 - z);
}

double primal_objective(vector<Instance*>* data, double* w, int D, double* alpha, int N, 
						double mu, double lambda, double lambda_2){
	
	double loss = 0.0;
	for(int i = 0; i < N; i++) {
		SparseVec xi = data->at(i)->xi;
		double yi = data->at(i)->yi;
		loss += smooth_hinge_loss(dot(w, xi) * yi, mu);
	}
	loss /= N;

	double obj_l2 = 0.0;
	for(int i = 0; i < D; i++) {
		obj_l2 +=  w[i]*w[i];
	}
	obj_l2 *= 0.5 * lambda_2;

	double obj_l1 = 0.0;
	for(int i = 0; i < D; i++) {
		obj_l1 += fabs(w[i]);
	}
	obj_l1 *= lambda;

	return loss + obj_l1 + obj_l2;
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
	
	int D;
	int N;
	vector<Instance*>* data =  Parser::parseSVM(trainFile,D);
	N = data->size();
	cerr << "N=" << N << endl; // # training data points
	cerr << "D=" << D << endl; // dimension

	//initialization
	double* v = new double[D];
	double* v_new = new double[D];
	double* u = new double[D];
	double* x_bar = new double[D];
	double* alpha = new double[N];

	double R = 0.0;
	for(int i = 0; i < N; i++) {
		SparseVec xi = data->at(i)->xi;
		double norm = 0.0;
		for (int k = 0; k < xi.size(); k++){	
			double value = xi[k].second;
			norm += value * value;
		}
		if (R < sqrt(norm)) {
			R = sqrt(norm);
		}
	}

	double tau = 0.5 / (R * sqrt(N));
	double sigma = 0.5 / R * sqrt(N);
	double theta = 1.0 - 1.0 / (double(N) + R * sqrt(N));

	lambda /= N; // L1 regularization constant
	double lambda_2 = 1.0 / N; // L2 regularization constant
	double mu = 1.0;

	cout << "tau: " << tau << endl;
	cout << "sigma: " << sigma << endl;
	cout << "theta: " << theta << endl;
	cout << "lambda: " << lambda << endl;
	cout << "lambda_2: " << lambda_2 << endl;
	cout << "mu: " << mu << endl;
	
	for(int i=0;i<D;i++) {
		v[i] = 0;
		u[i] = 0;
		x_bar[i] = 0;
	}
	
	for(int i=0;i<N;i++){
		alpha[i] = 0;
	}
	
	//Main Loop
	vector<int> index;
	for(int i = 0; i < N; i++) {
		index.push_back(i);
	}
	shuffle(index);

	int max_iter = 1000;
	int iter = 0;
	double nnz_v = 0.0;
	while(iter < max_iter){
		
		double update_time = -omp_get_wtime();
		for(int r = 0; r < N; r++) {
			
			int i = index[r];
			SparseVec xi = data->at(i)->xi;
			double yi = data->at(i)->yi;
			
			// update dual
			double new_alpha = (yi - dot(x_bar, xi) - (alpha[i] / sigma)) / (-1.0 - (1.0/sigma));
			if (yi > 0) {
				new_alpha = min( max( new_alpha, -1.0 ) , 0.0);
			} else {
				new_alpha = min( max( new_alpha, 0.0 ) , 1.0);
			}
			double alpha_diff = new_alpha-alpha[i];
			alpha[i] = new_alpha;
			// update primal
			for (int j = 0; j < D; ++j) {
				v_new[j] = (v[j] / tau - u[j]) / (lambda_2 + 1.0/tau);
			}
			for (int k = 0; k < xi.size(); k++){
				
				int idx = xi[k].first;
				double value = xi[k].second;
				
				v_new[idx] -= alpha_diff * value / (lambda_2 + 1.0/tau);
			}
			double threshold = lambda / (lambda_2 + 1.0/tau);
			for (int j = 0; j < D; ++j) {
				v_new[j] = prox_l1( v_new[j], threshold);
			}

			// maintain u
			for (int k = 0; k < xi.size(); ++k){
				
				int idx = xi[k].first;
				double value = xi[k].second;
				
				u[idx] += alpha_diff * value / N;
			}

			// maintain x_bar
			for (int j = 0; j < D; ++j) {
				x_bar[j] = v_new[j] + theta * (v_new[j] - v[j]);
				v[j] = v_new[j];
			}
		}
		update_time += omp_get_wtime();

		//if(iter%10==0)
		nnz_v = nnz(v, D);
		cerr << "iter=" << iter << ", nnz_a=" << nnz(alpha, N) 
		                        << ", nnz_v=" << nnz_v
		                        << ", obj=" << primal_objective(data, v, D, alpha, N, mu, lambda, lambda_2) 
		                        << ", time=" << update_time << endl ;
		
		shuffle(index);
		iter++;
	}
	cerr << endl;

	//output model
	ofstream fout(modelFile);
	fout << D << " " << nnz_v << endl;
	for(int i=0;i<D;i++)
		if( fabs(v[i]) > 1e-12 )
			fout << i << " " << v[i] << endl;
	fout.close();
}
