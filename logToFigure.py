import sys;
import os;

def parseDGPD(logPath):
    tripleList = list();
    with open(logPath) as fp:
       fp.readline(); 
       fp.readline();
       for line in fp:
           if len(line) > 30:
            tokens = line.split(', ');
            it = tokens[0].split('=')[1];
            obj = tokens[4].split('=')[1];
            time = tokens[5].split('=')[1];
            tripleList.append( (it,time,obj,) );
    return tripleList;

def parseDualRCD(logPath):
    tripleList = list();
    with open(logPath) as fp:
       fp.readline(); 
       fp.readline();
       for line in fp:
           if len(line) > 30:
            tokens = line.strip().split(', ');
            it = tokens[0].split('=')[1];
            obj = tokens[4].split('=')[1];
            time = tokens[5].split('=')[1];
            tripleList.append( (it,time,obj,) );
    return tripleList;

def parsePrimalRCD(logPath):
    tripleList = list();
    with open(logPath) as fp:
       fp.readline(); 
       fp.readline();
       for line in fp:
           if len(line) > 30:
            tokens = line.strip().split();
            it = tokens[0];
            obj = tokens[2];
            time = tokens[1];
            tripleList.append( (it,time,obj,) );
    return tripleList;

def parseSPDC(logPath):
    tripleList = list();
    with open(logPath) as fp:
       fp.readline(); 
       for line in fp:
           if len(line) > 30:
            tokens = line.strip().split(', ');
            it = tokens[0].split('=')[1];
            obj = tokens[3].split('=')[1];
            time = tokens[4].split('=')[1];
            tripleList.append( (it,time,obj,) );
    return tripleList;


def parseLog(logPath, method):
    if method =='DGPD':
        return parseDGPD(logPath);
    elif method == 'DualRCD':
        return parseDualRCD(logPath);
    elif method == 'PrimalRCD':
        return parsePrimalRCD(logPath);
    elif method == 'SPDC-dense':
        return parseSPDC(logPath);
    else:
        print('no parsing option matched for: ' + method);
        exit();

try:
    data_path = sys.argv[1];
    exec_path = sys.argv[2];
    output_dir = sys.argv[3];
except IndexError:
    print('script.py [data] [train_exec_path] [output_dir]');
    exit();


train_dir = exec_path[ : exec_path.rfind('/') ];
data_fname = data_path[ data_path.rfind('/')+1: ];

method = '';
if 'L1L2RSmoothHinge' in train_dir:
    method = 'DGPD';
elif 'DualRCD' in train_dir:
    method = 'DualRCD';
elif 'PrimalRCD' in train_dir:
    method = 'PrimalRCD';
elif 'SPDCDense' in train_dir:
    method = 'SPDC-dense';
else:
    print('no parsing option matched for: ' + logPath);
    exit();


for l2 in [0.1,0.01]:
    for l1 in [1,0.1,0.01]:
        log_path = train_dir+'/log.'+data_fname+'-l1-'+str(l1)+'-l2-'+str(l2);
        folder = output_dir+'/l2-'+str(l2)+'-l1-'+str(l1);
        os.system('mkdir '+ folder);
        os.system('mkdir '+ folder +'/iter-obj');
        os.system('mkdir '+ folder +'/time-obj');
        tripleList = parseLog( log_path, method );
        with open(folder+'/iter-obj/'+method,'w') as fp:
            fp.write('iter obj\n');
            for t in tripleList:
                fp.write(t[0]+' '+t[2]+'\n');
        with open(folder+'/time-obj/'+method,'w') as fp:
            fp.write('time obj\n');
            for t in tripleList:
                fp.write(t[1]+' '+t[2]+'\n');
