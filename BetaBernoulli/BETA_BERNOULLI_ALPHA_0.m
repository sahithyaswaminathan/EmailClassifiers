clc;
clear all;

%%Binarization of Train and Test data

load spamData.mat;

Xtrain_s = Xtrain;

for i = 1:3065 
    for j = 1:57  
    if Xtrain_s(i,j) > 0
        Xtrain_s(i,j) = 1;
    else
        Xtrain_s(i,j) = 0;
    end
    j=j+1;
    end
    i = i+1;
end

Xtest_s = Xtest;

for i = 1:1536 
    for j = 1:57  
    if Xtest_s(i,j) > 0
        Xtest_s(i,j) = 1;
    else
        Xtest_s(i,j) = 0;
    end
    j=j+1;
    end
    i = i+1;
end

%%Naive Bayes Beta-Bernoulli for alpha = 0

n1 = zeros(3065,57);

N = 3065;
yi = ytrain;
xij = Xtrain_s;

%%Calculating the Njc values

%Creating a matrix which stores value only when ytrain == 1 'spam'%
for i = 1:N
    c = yi(i,1);
    if c == 1 
        for j = 1:57
            n1(i,j) = xij(i,j);
        end
    end
end



N1 = n1(any(n1,2),:); %Deleting all the rows with the value zero%

Nj1 = sum(N1);

%Creating a matrix which stores value only when ytrain == 0 'not - spam'%
for i = 1:N
    c = yi(i,1);
    if c == 0 
        for j = 1:57
            n0(i,j) = xij(i,j);
        end
    end
end



N0 = n0(any(n0,2),:); %Deleting all the rows with the value zero%

Nj0 = sum(N0);

%%Calculating the class prior

pijc_1 = length(N1)/N; %prior maximum likelihood for the spam emails%
pijc_0 = length(N0)/N; %prior maximum likelihood for the non-spam email%

PI_1 = log (pijc_1);
PI_0 = log (pijc_0);


%% Calculating the theta jc values

for j = 1:57
       thetajc_1(1,j) = Nj1(1,j)/length(N1); %Theta jc for the spam category - length(N1)%
       thetajc_1(2,j) = Nj0(1,j)/length(N0); %Theta jc for the not-spam category - length (N0)%
end

L1_c0 = zeros(N,2);

%Calculating the prob for 'spam' class 1 [1*3065]%
for i = 1:N
    for j = 1:57
        if xij(i,j) == 1
            L1_c0(i,1) = L1_c0(i,1) + log (thetajc_1(1,j));
        else
            L1_c0(i,1) = L1_c0(i,1) + log (1- thetajc_1(1,j));
        end
    end
        prob(i,1) = PI_1 + L1_c0(i,1);
end

%Claculating the prob for 'not-spam' class 0 [1*3065]%
for i = 1:N
    for j = 1:57
        if xij(i,j) == 1
            L1_c0(i,2) = L1_c0(i,2) + log (thetajc_1(2,j));
        else
            L1_c0(i,2) = L1_c0(i,2) + log (1 - thetajc_1(2,j));
        end
    end
        prob(i,2) = PI_0 + L1_c0(i,2);
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
    if (label(i,1) ~= yi(i,1))
        testcount = testcount + 1;
    end
end
   
accuracy = (N - testcount) / N;

%% Calculating the probability for test values

xij_test = Xtest_s;

L1_c0_test = zeros(1536,2);

%Calculating the prob for 'spam' class 1 [1*3065]%
for i = 1:1536
    for j = 1:57
        if xij_test(i,j) == 1
            L1_c0_test(i,1) = L1_c0_test(i,1) + log (thetajc_1(1,j));
        else
            L1_c0_test(i,1) = L1_c0_test(i,1) + log (1- thetajc_1(1,j));
        end
    end
        prob_test(i,1) = PI_1 + L1_c0_test(i,1);
end

%Claculating the prob for 'not-spam' class 0 [1*3065]%
for i = 1:1536
    for j = 1:57
        if xij_test(i,j) == 1
            L1_c0_test(i,2) = L1_c0_test(i,2) + log (thetajc_1(2,j));
        else
            L1_c0_test(i,2) = L1_c0_test(i,2) + log (1 - thetajc_1(2,j));
        end
    end
        prob_test(i,2) = PI_0 + L1_c0_test(i,2);
end 

for i = 1:1536
if prob_test(i,1) > prob_test(i,2)
    label_t(i,1) = 1;
else
    label_t(i,1) = 0;
end
end

testcount_t= 0;

for i = 1:1536
    if (label_t(i,1) ~= ytest(i,1))
        testcount_t = testcount_t + 1;
    end
end
   
accuracy_test = (1536 - testcount_t) / 1536;







