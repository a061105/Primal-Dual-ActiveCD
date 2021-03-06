#include <algorithm>
#include <iomanip>
#include<cmath>
#include "loss.h"
#include "util.h"
#include "omp.h"
void rcd(vector<Feature*>& features, vector<double>& labels, int n, LossFunc* loss, double lambda, double lambda2,  double* w_ret, int nr_threads, int nIter, double stop_obj){
	double* factors = new double[n];
	
	int d = features.size();
	double k1 = loss->sec_deriv_ubound();//second derivative upper bound of the Loss
	double* H_diag = new double[d];
	for(int r=0;r<features.size();r++){
		Feature* fea = features[r];
		H_diag[r] = 0.0;
		for(SparseVec::iterator it=fea->values.begin(); it!=fea->values.end(); it++){
			H_diag[r] += it->second*it->second;
		}
		H_diag[r] *= k1;
		H_diag[r] += lambda2;
	}
	
	double* w = new double[d];
	for (int i=0;i<d; i++)
		w[i] = 0.0;
	double funval = 0.0;
	for (int i=0;i<n;i++){
		factors[i] = 0.0;
	}
	int max_iter = nIter;
	
	int ins_id;
	//double w_old = 0;
	double fval_new;
	int* pi = new int[d];
	for (int i=0;i<d;i++)
		pi[i] = i;
	
	int chunk = 10000;
	double start = omp_get_wtime();
	double minus_time=0.0;
	for (int iter=1;iter<=max_iter; iter++){
		
		random_shuffle(pi, pi+d);
		for (int inner_iter = 0; inner_iter < d; inner_iter++){
			
			int j = pi[inner_iter];
			Feature* fea = features[j];
			double Qii = H_diag[j];
			//update w
			double gradient =  0.0;
			for (SparseVec::iterator ii = fea->values.begin(); ii != fea->values.end(); ++ii){
				gradient += loss->deriv(factors[ii->first],labels[ii->first]) * ii->second;
			}
			gradient += lambda2*w[j];

			double eta = softThd(w[j] - gradient/Qii,lambda/Qii) - w[j];
			
			//maintain factors
			if( fabs(eta)>1e-10 ){
				w[j] += eta;
				for (SparseVec::iterator ii = fea->values.begin(); ii != fea->values.end(); ++ii){
					
					factors[ii->first] += eta * ii->second;
				}
			}
		}
		

		if ( iter % 10 == 0){

			minus_time -= omp_get_wtime();
			double L2term = 0.0;
			for(int j=0;j<d;j++){
				L2term += w[j]*w[j];
			}
			L2term *= lambda2/2.0;
			
			double L1term = 0.0;
			for(int j=0;j<d;j++){
				L1term += fabs(w[j]);
			}
			L1term *= lambda;
			
			double loss_term = 0.0;
			for (int i=0;i<n;i++){
				loss_term +=  loss->fval(factors[i],labels[i]);
			}
			
			double funval = L2term + L1term + loss_term;
			funval /= n;
			int nnz = 0;
			for (int i=0; i<d;i++){
				if (fabs(w[i])>1e-10)
					nnz++;
			}
			
			minus_time += omp_get_wtime();
			
			double end = omp_get_wtime();
			double time_used = end-start-minus_time;
			cerr  << setprecision(17)<<setw(20) << iter << setw(20) << time_used << setw(20) << funval << setw(10)<<nnz<< endl;
			//if( funval < stop_obj ){
			//	break;
			//}
		}
	}
	
	for(int i=0;i<d;i++){
		int j = features[i]->id;
		w_ret[j] = w[i];
	}

	delete [] H_diag;
	delete [] factors;
	delete [] w;
	delete [] pi;
}
