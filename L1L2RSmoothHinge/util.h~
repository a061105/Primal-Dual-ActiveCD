#ifndef UTIL_H
#define UTIL_H

#include<iostream>
#include<vector>
#include<deque>
#include<string>
#include<cmath>
#include<algorithm>
#include<unordered_map>

using namespace std;

typedef vector<pair<int,double> > SparseVec;
typedef unordered_map<int,double> HashVec;

double dot( double* v, SparseVec* x ){
	
	double sum=0.0;
	for(SparseVec::iterator it=x->begin(); it!=x->end(); it++){
		sum += v[it->first]*it->second;
	}
	
	return sum;
}

double dot( double* v, HashVec* x ){
	
	double sum=0.0;
	for(HashVec::iterator it=x->begin(); it!=x->end(); it++){
		sum += v[it->first]*it->second;
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

void transpose( vector<SparseVec*>& X, int N, int D, vector<SparseVec*>& Xt ){

	//clear
	for(int i=0;i<Xt.size();i++)
		delete Xt[i];
	Xt.clear();
	
	//initialize
	for(int j=0;j<D;j++)
		Xt.push_back(new SparseVec());
	//tranpose
	for(int i=0;i<N;i++){
		SparseVec* xi = X[i];
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			Xt[it->first]->push_back(make_pair(i,it->second));
		}
	}
}

class ScoreSmaller{
	
	public:
	ScoreSmaller(double* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] < score[ind2];
	}
	private:
	double* score;
};

class ScoreLarger{
	
	public:
	ScoreLarger(double* _score){
		score = _score;
	}
	bool operator()(const int& ind1, const int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	double* score;
};

#endif
