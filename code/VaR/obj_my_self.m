function obj = obj_my_self(p,x,y,tau)

yfit = yhat_my_self(p,x);
r = y-yfit;
obj=sum(abs(r.*(tau-(r<0))));