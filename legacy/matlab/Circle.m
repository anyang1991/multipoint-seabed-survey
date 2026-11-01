function 	Circle(x0,y0,r)
theta=0:pi/50:2*pi;
x=x0+r*cos(theta);
y=y0+r*sin(theta);
% plot(x,y,'-',x0,y0,'.');
fill(x,y,'y');
hold on;