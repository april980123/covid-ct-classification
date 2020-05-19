%%
covid1 = imread('CT_COVID/2019-novel-Coronavirus-severe-adult-respiratory-dist_2020_International-Jour-p3-89%0.png');
size(covid1)

%%

fid = fopen('trainCT_COVID.txt');
train_fname = textscan(fid,'%s');
fclose(fid);
train_name  = string(train_fname{:});
train_name 

size(train_name)

%%
covid_train_data = covid1 ;
for i = 1:191
    temp = imread('CT_COVID/'+train_name(i));
    size(temp)
end
