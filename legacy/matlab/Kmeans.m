function  [Stop,idx_Plus]=Kmeans(Target,Candidate_plus_stop, max_iters,Endurance_h,speed)
% globe Stop;

[~,n_stop]=size(Candidate_plus_stop);
[~,n_target]=size(Target);
X=[];
centroids=[];

for  i=1:n_target
    X=[X;Target(i).x,Target(i).y];
end

for  i=1:n_stop
    centroids=[centroids;Candidate_plus_stop(i).x,Candidate_plus_stop(i).y];
end


flag_ff=1;
while(flag_ff)
[centroids, idx] = runkMeans(X, centroids, max_iters,'ture');
flag_ff=0;

%判断是否所有点都在域内
[n,~]=size(X);
[k,~]=size(centroids);

for i=1:n
    Target(i).contain=0;
end

for i=1:k
    Center(i).x=centroids(i,1);
    Center(i).y=centroids(i,2);
    Center(i).contain=0;
end

for i=1:k
       for j=1:n
            if((Target(j).x-Center(i).x)^2+(Target(j).y-Center(i).y)^2<= (Endurance_h*speed/2)^2)
               Center(i).contain=1;
               break;
            end
       end
 end
 
count=1;
for i=1:k   
    if (Center(i).contain~=0)
        centroids(count,:)=[Center(i).x,Center(i).y];
        count=count+1;
    end
end

k=count-1;

 for i=1:n
    for j=1:k
        if((Target(i).x-Center(j).x)^2+(Target(i).y-Center(j).y)^2<=((Endurance_h-Target(i).workload)*speed/2)^2)
             Target(i).contain=1;
             break;
        end
    end
 end
Data_flag=0;
for i=1:n
    if (Target(i).contain==1)
        Data_flag=Data_flag+1;
    end
end
% Data_flag
if(Data_flag==n)
           flag_ff=1;
           clear T;
           T=Center;
           temp_idx=idx;
           % 找到最近的两个核去掉一个
           idx_plus=0;
           dis=Inf;
           for i=1:k-1
               for j=i+1:k
                    temp=(Center(i).x-Center(j).x)^2+(Center(i).y-Center(j).y)^2;
                    if temp<dis 
                        dis=temp;
                        idx_plus=i;
                    end
               end
           end

           Center(idx_plus)=[];
           k=k-1; 
           clear centroids;
           for i=1:k   
               centroids(i,:)=[Center(i).x,Center(i).y];
           end

else
    Stop=T;
    idx_Plus=temp_idx;    
    break;
           
end
end

