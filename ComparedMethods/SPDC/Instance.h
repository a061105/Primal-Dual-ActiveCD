#ifndef INSTANCE
#define INSTANCE

#include<vector>
#include<iostream>

using namespace std;

typedef vector<pair<int,double> > SparseVec;

class Instance{
	
	public:
	double yi;
	SparseVec xi;
	
	void print(){
		
		cout << yi << " ";
		/*for( int i=0;i<xi.size();i++){
			cout << xi[i].first << ":" << xi[i].second << " " ;
		}*/

		for( SparseVec::iterator it = xi.begin(); it!=xi.end(); it++)
			cout << it->first << ":" << it->second <<" ";
	}
};

#endif
