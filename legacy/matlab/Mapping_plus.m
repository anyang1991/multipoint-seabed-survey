%% 参数 

% 续航力 ：8h @ 7.2km/h


%% 宏变量
close all;
clc;
clear;

Endurance_h=8;                 %续航力
speed=7.2;
r_lim=Endurance_h*speed/2;               %极限圆半径
c_lim=r_lim*sqrt(2);             %极限正方形的边长

range=150;
n_target=15;

%% 生成位置点

% for i=1:n_target
%     Target(i).x= randi([ round(Endurance_h*speed/2), round(range-Endurance_h*speed/2)])+1;
%    
%     Target(i).y= randi([ round(Endurance_h*speed/2), round(range-Endurance_h*speed/2)])+1;
% 
%     Target(i).workload=randi([1,3]);
%     Target(i).sampling=randi([0,2]);
% end
load('temp');

%% 画点和连通域

figure('name','锚点候选区域') 
for i = 1:n_target
Circle(Target(i).x,Target(i).y,(Endurance_h-Target(i).workload)*speed/2);   %需要覆盖圆
hold on;
end  
for i = 1:n_target
    plot(Target(i).x,Target(i).y,'r*')
end  
axis([0 range+1 0 range+1]);

Tar_Matrix=zeros(range+1,range+1);
Tar_Matrix=logical(Tar_Matrix);

for i = 1:n_target

Tar_Matrix(Target(i).x,Target(i).y)=1;
      
      for x_draw=floor(-Endurance_h*speed/2):ceil(Endurance_h*speed/2)
           for y_draw=floor(-Endurance_h*speed/2):ceil(Endurance_h*speed/2)
               if(x_draw^2+y_draw^2)<=((Endurance_h-Target(i).workload)*speed/2)^2
                    Tar_Matrix(Target(i).x+x_draw,Target(i).y+y_draw)=1;
               end
           end
      end
end
% % 画矩阵形式
% figure
% imshow(Tar_Matrix);



%% 测量连通域
Tar_Matrix=bwlabel(Tar_Matrix,8);
STATS=regionprops(Tar_Matrix,'Centroid','BoundingBox');

[n_STATS,~]=size(STATS);
temp=zeros(n_STATS,1);

for i=1:n_STATS

    temp=STATS(i).Centroid(1);
    STATS(i).Centroid(1)=STATS(i).Centroid(2);
    STATS(i).Centroid(2)=temp;
  
    temp=STATS(i).BoundingBox(1);
    STATS(i).BoundingBox(1)=STATS(i).BoundingBox(2);
    STATS(i).BoundingBox(2)=temp;
    
    temp=STATS(i).BoundingBox(3);
    STATS(i).BoundingBox(3)=STATS(i).BoundingBox(4);
    STATS(i).BoundingBox(4)=temp;
end

for i = 1:n_target                               %分类
   Target(i).idx=Tar_Matrix(Target(i).x,Target(i).y);
end

 for i=1:n_STATS
    Area(i).x=ceil((STATS(i).BoundingBox(3)/(Endurance_h*speed/2)));
    Area(i).y=ceil((STATS(i).BoundingBox(4)/(Endurance_h*speed/2)));
 end

% %% 显示连通域中心
% figure
% for i=1:n_STATS
% Circle(STATS(i).Centroid(1),STATS(i).Centroid(2),2);
% end
% axis([0 range+1 0 range+1]);
% hold on;
% for i = 1:n_target
%     plot(Target(i).x,Target(i).y,'r*');
% end

%% 归类连通域
Big_Area_count=1;
Small_Area_count=1;

for i=1:n_STATS
        Big_Area(Big_Area_count).Centroid=STATS(i).Centroid;
        Big_Area(Big_Area_count).BoundingBox=STATS(i).BoundingBox;
        Big_Area(Big_Area_count).x=Area(i).x;
        Big_Area(Big_Area_count).y=Area(i).y;
        Big_Area(Big_Area_count).idx=i;
        Big_Area_count=Big_Area_count+1;
end
Big_Area_count=Big_Area_count-1;


%% 大区域候选车站
Candidate_stop_count=1;
for i=1:Big_Area_count
    for j=1:ceil(Big_Area(i).BoundingBox(3)/c_lim)
        for k=1:ceil(Big_Area(i).BoundingBox(4)/c_lim)
            Candidate_stop(Candidate_stop_count).x=Big_Area(i).BoundingBox(1)+c_lim*(j-0.5);
            Candidate_stop(Candidate_stop_count).y=Big_Area(i).BoundingBox(2)+c_lim*(k-0.5);
            Candidate_stop(Candidate_stop_count).idx=Big_Area(i).idx;
            Candidate_stop_count=Candidate_stop_count+1;
        end
    end
end
Candidate_stop_count=Candidate_stop_count-1;

%% 画大区域候选车站
figure('name','保证覆盖的锚点范围') 
for i = 1:n_target
    Circle(Target(i).x,Target(i).y,(Endurance_h-Target(i).workload)*speed/2);   %需要覆盖圆
    hold on;
    plot(Target(i).x,Target(i).y,'r*')
end
for i=1:Candidate_stop_count
    Rectangle(Candidate_stop(i).x,Candidate_stop(i).y,c_lim);
end
axis([0 range+1 0 range+1]);

%% 大区域候选车站处理
Candidate_plus_stop_count=1;
for i=1:Candidate_stop_count
    flag=0; 
        for j = 1:n_target
            if Target(j).idx==Candidate_stop(i).idx
                if (abs(Target(j).x-Candidate_stop(i).x)<c_lim/2)&&(abs(Target(j).y-Candidate_stop(i).y)<c_lim/2) 
                    flag=1;
                    break;
                end
            end
        end
       if (flag==1)
                Candidate_plus_stop(Candidate_plus_stop_count).x=Candidate_stop(i).x;
                Candidate_plus_stop(Candidate_plus_stop_count).y=Candidate_stop(i).y;               
                Candidate_plus_stop(Candidate_plus_stop_count).idx=Candidate_stop(i).idx;
                Candidate_plus_stop_count=Candidate_plus_stop_count+1;
       end
end
Candidate_plus_stop_count=Candidate_plus_stop_count-1;
       
%% 画大区域过滤后车站
figure('name','粗选锚点范围') 


for i = 1:n_target
    Circle(Target(i).x,Target(i).y,(Endurance_h-Target(i).workload)*speed/2);   %需要覆盖圆
end  
for i=1:Candidate_plus_stop_count
    Rectangle(Candidate_plus_stop(i).x,Candidate_plus_stop(i).y,c_lim);
end
for i = 1:n_target
    plot(Target(i).x,Target(i).y,'r*')
end  
axis([0 range+1 0 range+1]);


% clearvars -EXCEPT Target Candidate_plus_stop Endurance_h speed n_target
save('temp.mat');


