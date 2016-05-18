import sys;
import os;

#./parallelRCD [data] [testdata] [L1_lambda] [L2_lambda] [loss(0:square,1:L2-hinge,2:logistic,3:SmoothHinge)] [num_threads] [num_iter]

try:
    data_path = sys.argv[1];
    test_data_path = sys.argv[2];
except IndexError:
    print('script2.py [data] [test_data]');
    exit();

exec_path = 'ComparedMethods/PrimalRCD/parallelRCD';

train_dir = exec_path[ : exec_path.rfind('/') ];
data_fname = data_path[ data_path.rfind('/')+1: ];

for l2 in [0.1,0.01]:
    for l1 in [1,0.1,0.01]:
        log_path = train_dir+'/log.'+data_fname+'-l1-'+str(l1)+'-l2-'+str(l2);
        os.system(exec_path+' '+data_path+' '+test_data_path+' '+str(l1)+' '+str(l2)+' 3 1 1000 > '+log_path+' 2> '+log_path);
