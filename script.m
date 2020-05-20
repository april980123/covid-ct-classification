%%
covid1 = imread('CT_COVID/2020.03.22.20040782-p25-1542.png');
size(covid1)

%%

fid = fopen('testCT_COVID.txt');
train_fname = textscan(fid,'%s');
fclose(fid);
train_name  = string(train_fname{:});
train_name 

size(train_name)

%%
first_pic = covid1 ;
covid1 = rgb2gray(covid1);
s = size(covid1);
%the largest dimension of the nonCOVID train pictures are 830 and 797
for i = s(1):1225
    for j = s(2):1671
            covid1(i,j) = 0;
    end
end

%%
test_COVID=covid1;
test_COVID(1,400)
size(covid1)

%%
for i = 2:98
    temp = imread('CT_COVID/'+train_name(i));
    if numel(size(temp))>=3
        temp = rgb2gray(temp);
    end
    t = size(temp);
    if t(1) < 1225 && t(2) < 1671
        for z = t(1):1225
            for j = t(2):1671
                temp(z,j) = 0;
            end
        end
        for j = t(2):1671
            for z = t(1):1225
                temp(z,j) = 0;
            end
        end
    end
    if t(2) < 1671 && t(1) == 1225
        for j = t(2):1671
                temp(:,j) = 0;
        end  
    end
    if t(1) < 1225 && t(2) ==1671
        for z = t(1):1225
                temp(z,:) = 0;
        end  
    end
%     if t(1) > s(1)
%         s(1) = t(1);
%     end
%     if t(2) > s(2)
%         s(2) = t(2);
%     end
    %size(temp)
    test_COVID = [test_COVID temp];
end


%%
save('test_COVID.mat','test_COVID')

%%
clear variables
load('train_NonCOVID.mat')
load('train_COVID.mat')
load('test_NonCOVID.mat')
load('test_COVID.mat')


