
clear;

data = importdata('E:\matlabÎÄ¼ş\VaR\indices_since95.csv');
indices = data.data(:,2);
indices = indices(1:end-1);


y = log(indices(2:end))-log(indices(1:end-1));

x = [abs(y(1:end-1)).*(y(1:end-1)>0),abs(y(1:end-1)).*(y(1:end-1)<0)];

% %[p,stat] = quantreg_my_self(x,y(2:end),0.1,1,200);
% [p,stat] = quantreg(x,y(2:end),0.05,1,200);
% 
% p'./stat.pse
% p'

y0 = y(2:end);
p = quantreg_my_self(x,y0,0.1,1,200);

xf = x;
xf(:,2:end+1) = x(:,1:end);
xf(:,1) = 1;
xf(:,end+1) = 0;

yfit = yhat_my_self(p,xf);

figure, plot(y0);
hold on, plot(yfit,'r'), hold off;


