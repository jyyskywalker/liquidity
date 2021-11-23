function y = yhat_my_self(p,x)

y = zeros(size(x,1),1);
x(1,end) = p(end);
y(1) = x(1,:)*p(1:end-1);
for ii=2:size(x,1)
    x(ii,end) = y(ii-1);
    y(ii) = x(ii,:)*p(1:end-1);
end