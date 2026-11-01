clear; clc

%%

load('temp.mat');

% Settings for running K-Means
max_iters = 100;


[Stop,idx]=Kmeans(Target,Candidate_plus_stop, max_iters,Endurance_h,speed);
[~,n_Stop]=size(Stop);

% 画出大区域最终车站
figure('name','最终锚点范围') 
for i=1:n_Stop
Circle(Stop(i).x,Stop(i).y,Endurance_h*speed/2);
end
hold on;
for i=1:n_target
   plot(Target(i).x,Target(i).y,'r*')
end
[M,N]=hist(idx,unique(idx));


dis=Inf;
for i=1:n_target
   for j=1:n_Stop
       if (Target(i).x-Stop(j).x)^2+ (Target(i).y-Stop(j).y)^2 < dis
           dis=(Target(i).x-Stop(j).x)^2+ (Target(i).y-Stop(j).y)^2;
           Target(i).belong=j;
       end
     end
  dis=Inf;      
end

for i=1:n_Stop
    Stop(i).include=0;
end
for i=1:n_target
    Stop(Target(i).belong).include= Stop(Target(i).belong).include+1;
end

max_include=0;

for i=1:n_Stop
    if Stop(i).include>max_include
        max_include=Stop(i).include;
        indx=i;
    end
end
% 
% indx=1; %规定锚点

the_Stop=Stop(indx);

j=1;
for i=1:n_target
    if Target(i).belong==indx
       the_Target(j)=Target(i);
       j=j+1;
    end
end



clearvars -EXCEPT the_Target the_Stop Endurance_h speed
save('temp2.mat');


