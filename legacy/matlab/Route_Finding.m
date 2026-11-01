clear;
clc;
load ('temp2.mat');
%%  参数区
% Stop                    %n个车站的坐标，n×2的矩阵
% Load                    %车辆内装载量 
% Loopcount_max           %最大迭代次数
% n_ant                   %蚂蚁个数
% K_Phe                   %表征信息素重要程度的参数
% K_Dis                   %表征距离重要程度的参数
% K_Load                  %表征装载量重要程度的参数
% Rho_0                   %初始信息素蒸发系数
% Rho_max                 %最大信息素蒸发系数
% Q                       %信息素增加强度系数
% R_Best                  %各代最佳路线
% L_Best                  %各代最佳路线的长度
% L_ave                   %各代路线的平均长度
% Pheromone               %费洛蒙矩阵
% Phe_max                 %费洛蒙最大值
% Phe_min                 %费洛蒙最小值
% Route                   %禁忌表
% Distance                %站间距离矩阵
% Distance_0              %站距学校矩阵
% Delta_Pheromone         %费洛蒙变化量矩阵
% School                  %学校坐标 
 
%% PART I      变量初始化
Load=3;



K_Phe=1;                  
K_Dis=1;                  
K_Load=1;                  
Loopcount_max=200;           
n_ant=30; 
Rho=0.5;
Q=1.5;                     
            
Stop=the_Stop;

Target=the_Target;
[~,n]=size(Target);


%站间距离矩阵
Distance=zeros(n,n);
for i=1:n
    for j=1:n
         if i~=j
             Distance(i,j)=sqrt((Target(i).x-Target(j).x)^2+(Target(i).y-Target(j).y)^2);
         else
             Distance(i,j)=eps;      %i=j时不计算，应该为0，但后面的启发因子要取倒数，用eps（浮点相对精度）表示
         end
    end
end

%每点距学校距离 
Distance_0=zeros(n,1); 
for i=1:n
    Distance_0(i,1)=sqrt((Target(i).x-Stop.x)^2+(Target(i).y-Stop.y)^2);
end

Length_best=inf.*ones(Loopcount_max,1);   
Length_average=zeros(Loopcount_max,1);        
Pheromone=ones(n,n);               
Delta_Pheromone=zeros(n,n);   
Dis_inv=1./Distance; 
ant_load=zeros(n_ant,1);
Stop_unVisited=ones(n,1);
Dis_inv_0=1./Distance_0; 

Dis_inv_0=(Dis_inv_0-min(min(Dis_inv_0)))/(max(max(Dis_inv_0))-min(min(Dis_inv_0)));


Dis_inv(i,j)=0; 
Dis_inv=(Dis_inv-min(min(Dis_inv)))/(max(max(Dis_inv))-min(min(Dis_inv)));



Loopcount=1;
while (Loopcount<Loopcount_max)
%% PART II  放置蚂蚁和完成周游
%所有蚂蚁置于初始点
clear Route;
for i=1:n_ant
   Route{i}(1)=0;         %0代表是船
   ant_endurance(i)=Endurance_h*speed;
end


%完成周游
for i=1:n_ant   %i是蚂蚁编号
% i=1;
    j=2;
       while (sum(Stop_unVisited)~=0)  %如果没有都访问过
           n_unVisited=1; %待访问列表的初值
           %记录已访问    
            Visited=(Route{i}(1:(j-1))); %Visited是访问过的列表
            for k=1:length(Route{i})    
               if Route{i}(k)>0          %如果不是船
                  Stop_unVisited(Route{i}(k))=0; %那么这个站访问过了 访问过是0
               end
            end
            
            clear J; %J是带访问列表
            
            %待访问列表
              for k=1:n
                if((Stop_unVisited(k)==1))
                    J(n_unVisited)=k;
                    n_unVisited=n_unVisited+1;
                end
              end 
              
            %不超载且可回船列表
            Count_J=1;
            Temp_J=[];
             for k=1:n_unVisited-1
               if Visited(end)~=0;          %如果现在不在船上 那就要看满不满足条件
                  if (ant_load(i)+Target(J(k)).sampling)<=Load && (ant_endurance(i)-Distance_0(J(k))- Distance(Visited(end),J(k)) )>=0
                     Temp_J(Count_J)=J(k);
                     Count_J=Count_J+1;                
                  end
               else                        %如果在船上，没去过的地方都行
                     Temp_J=J;
   
               end
             end
             
            %若列表为空
            if length(Temp_J)==0
             Route{i}(j)=0; 
             j=j+1;
             Visited(end+1)=0;
             ant_load(i)=0;
             ant_endurance(i)=Endurance_h*speed; 
             continue;
            end         
%这地方改动了
            
            
            %计算待选城市概率分布          
            P=Temp_J;
            for k=1:length(Temp_J)
                if(Visited(end)==0)
                       P(k)= (0.1^K_Phe)*(1/Distance_0(Temp_J(k))^K_Dis);       
                else
                     if Temp_J(k)~=0;
                         P(k)=(Pheromone(Visited(end),Temp_J(k))^K_Phe)*(Dis_inv(Visited(end),Temp_J(k))^K_Dis);
                     else
                         P(k)=(0.1^K_Phe)*(1/Distance_0(Visited(end))^K_Dis) ;
                     end
                end
            end
            
            P=P/(sum(P));
            %按概率选择下一城市
            Pcum=cumsum(P);
            Select=find(Pcum>rand);
            Route{i}(j)=Temp_J(Select(1));
            
            if Route{i}(j)~=0
                if Visited(end)~=0
                    ant_load(i)=ant_load(i)+Target(Temp_J(Select(1))).sampling;
                    ant_endurance(i)=ant_endurance(i)-Distance(Temp_J(Select(1)),Visited(end));
                    Stop_unVisited(Temp_J(Select(1)))=0;
                else
                    ant_load(i)=ant_load(i)+Target(Temp_J(Select(1))).sampling;
                    ant_endurance(i)=ant_endurance(i)-Distance_0(Temp_J(Select(1)));
                    Stop_unVisited(Temp_J(Select(1)))=0;
                end
             else
                
                ant_load(i)=0;
                ant_endurance(i)=Endurance_h*speed;
                
            end
             j=j+1;
       end
       
       Stop_unVisited=ones(n,1); %每只蚂蚁走完后重置初始列表
end



  for k=1:n_ant
      if Route{k}(end)~=0;
         Route{k}(end+1)=0;
         ant_load(i)=0;
         ant_endurance(i)=Endurance_h*speed;
      end
  end
  
 if Loopcount>=2  %带有下标的赋值维度不匹配。
  Route{1}=[];
  Route{1}=R_Best{Loopcount-1};
 end
  %% PART III  确定本次迭代最佳路线

  L=zeros(n_ant,1);
  for i=1:n_ant
      R=Route{i};
      Current=0;
      MIN_R(i)=Inf;
      MAX_R(i)=1;
      for j=1:(length(R)-1)
          if ((R(j)~=0)&&(R(j+1)~=0))
              L(i)=L(i)+Distance(R(j),R(j+1));
              Current=Current+Distance(R(j),R(j+1));
          elseif (R(j)+R(j+1)~=0)
              L(i)=L(i)+Distance_0(R(j)+R(j+1));
              Current=Current+Distance_0(R(j)+R(j+1));
               if R(j+1)==0
                  if Current>MAX_R(i)  
                      MAX_R(i)=Current;
                  end
                   if Current<MIN_R(i)  
                      MIN_R(i)=Current;
                   end
                   Current=0;
              end      
          else continue;
          end
      end
  end
%   L=L.*(((MAX_R')./(MIN_R')).^0.25);
  L_Best(Loopcount)=min(L);
  pos=find(L==L_Best(Loopcount));
  R_Best{Loopcount}=Route{pos(1)}; %带有下标的赋值维度不匹配。
  L_ave(Loopcount)=mean(L);          
  Loopcount=Loopcount+1
  %% PART IV 更新信息素
  
    Delta_Pheromone=zeros(n,n);        
    for i=1:n_ant
        for j=1:length(Route{i})-1;
            if((Route{i}(j)~=0)&&(Route{i}(j+1)~=0))
                Delta_Pheromone(Route{i}(j),Route{i}(j+1))=Delta_Pheromone(Route{i}(j),Route{i}(j+1))+Q/L(i);  
            end
        end
      end
     Pheromone=(1-Rho).*Pheromone+Delta_Pheromone; %考虑信息素挥发，更新后的信息素
     Pheromone=(Pheromone-min(min(Pheromone)))/(max(max(Pheromone))-min(min(Pheromone)))+1;
   
 end

Pos=find(L_Best==min(L_Best)); %找到最佳路径（非0为真）
Shortest_Route=R_Best{Pos(1)} ;%最大迭代次数后最佳路径
Shortest_Length=L_Best(Pos(1)); %最大迭代次数后最短距离


