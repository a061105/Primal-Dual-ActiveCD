w = load('model');
index = find(w~=0);

fp = fopen('selectIndex','w');
fprintf(fp,'%d\n',index);
fclose(fp);
