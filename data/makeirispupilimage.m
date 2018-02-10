fn = dir('mask/');
fname = [];

for i=1:size(fn,1)
    fname = strcat('mask/',fn(i).name);
    fname2 = strcat('mask_iris/',fn(i).name);

    if fn(i).isdir == 1
        continue; 
    end
    
    fname2
    
    if exist(fname,'file') == 2 && exist(fname2,'file') == 2
        fprintf(1,'%s\n',fname);
    else
        continue;
    end
    
    I = imread(fname);
    figure(1), imshow(I);
    I2 = imread(fname2);
    figure(2), imshow(I2);
    pause;

end
