import java.io.*;
import java.util.*;

class FeaExtract{
	
	static Set<Integer> nnzFeas = new HashSet();

	public static void main(String[] args){
		
		if( args.length < 2 ){
			System.err.println("Usage: java FeaExtract [data] [fea_selected]");
			System.exit(0);
		}
		
		String dataFile = args[0];
		String feaIndexFile = args[1];

		readFeaIndex(feaIndexFile);
		writeNewData(dataFile);
	}
	
	static void writeNewData(String datafile){
		
		try{
			BufferedReader bufr = new BufferedReader(new FileReader(datafile));
			BufferedWriter bufw = new BufferedWriter(new FileWriter(datafile+".nzf"));
			
			String line;
			String[] tokens;

			while( (line=bufr.readLine()) != null ){

				tokens = line.split(" ");
				bufw.write(tokens[0]);
				for(int i=1;i<tokens.length;i++){
					
					Integer index = Integer.valueOf(tokens[i].split(":")[0]);
					if( nnzFeas.contains(index) ){
						bufw.write( " "+tokens[i] );
					}
				}
				bufw.newLine();
			}
			bufw.close();
			bufr.close();

		}catch(Exception e){
			e.printStackTrace();
			System.exit(0);
		}
	}

	static void readFeaIndex(String feaFile){
		
		try{
			BufferedReader bufr = new BufferedReader(new FileReader(feaFile));
			String line;
			while( (line=bufr.readLine()) != null ){
				
				nnzFeas.add(Integer.valueOf(line));
			}
			bufr.close();
			
		}catch(Exception e){
			e.printStackTrace();
			System.exit(0);
		}
	}
}
