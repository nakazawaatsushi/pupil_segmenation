fn = dir('mask/');
fname = [];

for i=1:size(fname,1)
    fname(i) = strcat('mask/',fn(i));
    fprintf(1,'%s\n',fname(i).name);
    if fname(i).isdir == 1
        continue; 
    end
    
    fprintf(1,'%s\n',fname(i).name);
    
    I = imread(fname(i),0);
    M = mean(I);
    imshow(I);
    hold on;
    plot(M(1),M(2),'x');
    hold off;
    pause;

end
