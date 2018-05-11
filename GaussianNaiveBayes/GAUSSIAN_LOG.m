clc;

clear all;

load spamData.mat;

%%Gaussian Naive Bayes for the Log normalization

log1 = zeros(3065,57);

N = 3065;

yl = ytrain;
xlog = log(Xtrain+0.1);

%Creating a matrix which stores value only when ytrain == 1 'spam'%
for i = 1:N
    c = yl(i,1);
    if c == 1 
        for j = 1:57
            log1(i,j) = xlog(i,j);
        end
    end
end

t1 = log1(any(log1,2),:); %Deleting all the rows with the value zero%

T1 = sum(t1);

for j = 1:57
    mu_1(1,j) = T1 (1,j) / 1234;
end

for i = 1: length(t1)
    for j = 1:57
    var_0(i,j) =(t1(i,j) - mu_1(1,j))^2/length(t1);
    end
end
var_1(1,:) = sum(var_0);
    
%Creating a matrix which stores value only when ytrain == 0 'not - spam'%
for i = 1:N
    c = yl(i,1);
    if c == 0 
        for j = 1:57
            log0(i,j) = xlog(i,j);
        end
    end
end



t0 = log0(any(log0,2),:); %Deleting all the rows with the value zero%
    
T0 = sum(t0);

for j = 1:57
    mu_1(2,j) = T0 (1,j)/1831; %Calculating the mean%
end
for i = 1:length(t0)
    for j = 1:57
    var_00(i,j) = (t0(i,j) - mu_1(2,j))^2/length(t0); %Calculating the variance%
    end
end
var_1(2,:) = sum(var_00);


%%Calculating the class prior

pijc_1 = length(T1)/N; %prior maximum likelihood for the spam emails%
pijc_0 = length(T0)/N; %prior maximum likelihood for the non-spam email%

PI_1 = log (pijc_1);
PI_0 = log (pijc_0);

%%Calculating the probability

L1_l0 = zeros(N,2);

%Calculating the prob for 'spam' class 1 [1*3065]%
for i = 1:N
    for j = 1:57
            L1_l0(i,1) = L1_l0(i,1) - ((xlog(i,j)-mu_1(1,j))^2/(2*var_1(1,j))) - log(sqrt(2*pi*var_1(1,j)));
    end
    prob(i,1) = PI_1 + L1_l0(i,1);
end
       
for i = 1:N
    for j = 1:57
        L1_l0(i,2) = L1_l0(i,2) - ((xlog(i,j)-mu_1(2,j))^2/(2*var_1(2,j))) - log(sqrt(2*pi*var_1(2,j)));
    end
    prob(i,2) = PI_0 + L1_l0(i,2);
end

for i = 1:N
if prob(i,1) > prob(i,2)
    label(i,1) = 1;
else
    label(i,1) = 0;
end
end

testcount= 0;

for i = 1:N
    if (label(i,1) ~= ytrain(i,1)) %Counting the mis-classified value%
        testcount = testcount + 1;
    end
end
   
accuracy = (N - testcount) / N;
error = (1-accuracy)*100;
    
    
%%Calculating the test accuracy and probability

L1_t0 = zeros(1536,2);

xtest = log(Xtest+0.1);

for i = 1:1536
    for j = 1:57
            L1_t0(i,1) = L1_t0(i,1) - ((xtest(i,j)-mu_1(1,j))^2/(2*var_1(1,j))) - log(sqrt(2*pi*var_1(1,j)));
    end
    prob_test(i,1) = PI_1 + L1_t0(i,1);
end
       
for i = 1:1536
    for j = 1:57
        L1_t0(i,2) = L1_t0(i,2) - ((xtest(i,j)-mu_1(2,j))^2/(2*var_1(2,j))) - log(sqrt(2*pi*var_1(2,j)));
    end
    prob_test(i,2) = PI_0 + L1_t0(i,2);
end

for i = 1:1536
if prob_test(i,1) > prob_test(i,2)
    label_test(i,1) = 1;
else
    label_test(i,1) = 0;
end
end

count= 0;

for i = 1:1536
    if (label_test(i,1) ~= ytest(i,1)) %Counting the mis-classified values%
        count = count + 1;
    end
end
   
accuracy_test = (1536 - count) / 1536;    
error_test = (1-accuracy_test)*100;    
