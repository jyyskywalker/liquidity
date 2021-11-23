% stock_list = ["000001.SZ","600015.SH","601398.SH","601939.SH","601988.SH","601998.SH"];
% 
token = '094f15d71394516b730602faa77b1c708007b8d05df300590b4445ed';
api = pro_api(token);

stock_list = {};
% 从 txt 文件读取stock_list
path = 'E:\matlab文件\VaR\stock_list.txt';
fid = fopen(path,'r');
tline = fgetl(fid);
while  ischar(tline)
    stock_list{end+1} = tline;
    tline = fgetl(fid);
end

% stock_list = {'000001.SZ','600015.SH','601398.SH','601939.SH','601988.SH','601998.SH'};
% stock_list = ["000001.SZ","600015.SH","601398.SH","601939.SH","601988.SH","601998.SH"];
start_date = "20180101";
end_date = "20201231";
df_daily = table;
df_date = table;
for i = 1:size(stock_list,2)
    temp = api.query('daily','ts_code',stock_list{i},'start_date',start_date,'end_date',end_date);
    % a = ['df_daily.',stock_list{i}, '=', 'log(temp.close)-log(temp.pre_close)'];
    % eval(a)
    % df_daily(:,i) = table(log(temp.close)-log(temp.pre_close));
    df_daily(:,i) = table(flipud(temp.close));
    df_date(:,1) = table(flipud(temp.trade_date(1:end-2)));
end


% 计算 VaR 风险度量

prob = 0.1;
var_lag = 1;
iter_num = 200;

VaR = table;
log_yield=table;
for i = 1:size(stock_list,2)
    indices = table2array(df_daily(:,i));
    y = log(indices(2:end))-log(indices(1:end-1));
    x = [abs(y(1:end-1)).*(y(1:end-1)>0),abs(y(1:end-1)).*(y(1:end-1)<0)];
    y0 = y(2:end);
    p = quantreg_my_self(x,y0,prob,var_lag,iter_num);
    xf = x;
    xf(:,2:end+1) = x(:,1:end);
    xf(:,1) = 1;
    xf(:,end+1) = 0;

    yfit = yhat_my_self(p,xf);
    VaR(:,i) = table(yfit);
    log_yield(:,i) = table(y0);
end
VaR.Properties.RowNames = df_date.Var1;
log_yield.Properties.RowNames = df_date.Var1;
writetable(VaR,'E:\matlab文件\VaR\VaR.csv','Delimiter',',','WriteRowNames',true);
writetable(log_yield,'E:\matlab文件\VaR\log_yield.csv','Delimiter',',','WriteRowNames',true);