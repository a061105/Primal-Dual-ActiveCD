
#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<vector>
#include<map>
#include<iostream>
#include<cstring>
#include "util.h"

using namespace std;


const int MAX_LINE = 10000000;

class Parser{
	
	public:
	static  void  parseSVM(char* fileName, int& numFea, vector<SparseVec*>& X, vector<double>& labels){
		
		ifstream fin(fileName);
		if( fin.fail() ){
			cerr << "cannot find file." << endl;
			exit(0);
		}
		
		labels.clear();
		char* line = new char[MAX_LINE];
		vector<string> tokens;
		numFea=0;
		while( !fin.eof() ){
			
			SparseVec* xi = new SparseVec();
			
			fin.getline(line,MAX_LINE);
			string str = string(line);
			tokens = split(str," ");
			if( str.length() < 2 )
				continue;

			//yi
			double label = atof(tokens[0].c_str());
			labels.push_back(label);
			
			//xi
			for(int i=1;i<tokens.size();i++){
				
				vector<string> pv = split(tokens[i],":");
				pair<int,double> pair;
				pair.first = atoi(pv[0].c_str());
				pair.second = atof(pv[1].c_str());
				xi->push_back(pair);
			}
			//cerr << "fea="<< ins->xi.back().second << endl;
			//cerr << data->size() << ", " << ins->xi.size() <<  endl;
			if( xi->size()>0 && xi->back().first > numFea )
				numFea = xi->back().first;
			
			X.push_back( xi );
		}
		
		//data->pop_back();
		numFea++;

		delete[] line;
	}

	static vector<string> split(string str, string pattern){

		vector<string> str_split;
		size_t i=0;
		size_t index=0;
		while( index != string::npos ){

			index = str.find(pattern,i);
			str_split.push_back(str.substr(i,index-i));

			i = index+1;
		}
		
		if( str_split.back()=="" )
			str_split.pop_back();
		
		return str_split;
	}

};
