
#include"Instance.h"
#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<vector>
#include<map>
#include<iostream>
#include<cstring>


using namespace std;

const int MAX_LINE = 10000000;

class Parser{
	
	public:
	static  vector< Instance* >*  parseSVM(char* fileName, int& numFea){
		
		ifstream fin(fileName);
		if( fin.fail() ){
			cerr << "cannot find file." << endl;
			exit(0);
		}
		
		vector< Instance* >* data = new vector< Instance* >();
		
		char* line = new char[MAX_LINE];
		vector<string> tokens;
		numFea=0;
		while( !fin.eof() ){
			
			Instance* ins = new Instance();
			
			fin.getline(line,MAX_LINE);
			string str = string(line);
			tokens = split(str," ");
			if( str.length() < 2 )
				continue;

			//yi
			ins->yi = atof(tokens[0].c_str());

			//xi
			for(int i=1;i<tokens.size();i++){
				
				vector<string> pv = split(tokens[i],":");
				pair<int,double> pair;
				pair.first = atoi(pv[0].c_str());
				pair.second = atof(pv[1].c_str());
				ins->xi.push_back(pair);
			}
			//cerr << "fea="<< ins->xi.back().second << endl;
			//cerr << data->size() << ", " << ins->xi.size() <<  endl;
			if( ins->xi.size()>0 && ins->xi.back().first > numFea )
				numFea = ins->xi.back().first;
			
			data->push_back( ins );
		}
		
		//data->pop_back();
		numFea++;
		delete[] line;
		return data;
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
