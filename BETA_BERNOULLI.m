clc;
clear all;

%%Binarization of Train and Test data

load spamData.mat;

Xtrain_s = Xtrain;

for i = 1:3065 
    for j = 1:57  
    if Xtrain_s(i,j) > 0 %Values greater than zero are 
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

%%Naive Bayes Beta - bernoulli for the alpha values = 1 to 100

N = 3065;
yi = ytrain;
xij = Xtrain_s;

n1 = zeros(3065,57);

%Creating a matrix which stores value only when ytrain == 0 'not - spam'%
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

%% Calculating theta and prob for alpha = 1 to 100

L1_a = zeros(N,200);
L0_a = zeros(N,200);

 for j = 1:57
     a = 1;
     if a <= 100
        for k = 1 : 200
            thetajc_1a(k,j) = (Nj1(1,j)+a)/(length(N1) + 2 *a); %Theta jc for the spam category - length(N1)%
            thetajc_0a(k,j) = (Nj0(1,j)+a)/(length(N0) + 2 *a);%Theta jc for the not-spam category - length (N0)%
            a = a +0.5;
        end
    end    
 end

for i = 1:N
    for j = 1:57
        for k = 1 : 200
            if xij(i,j) == 1
                L1_a(i,k) = L1_a(i,k) + log (thetajc_1a(k,j));
            else
                L1_a(i,k) = L1_a(i,k) + log (1- thetajc_1a(k,j));
            end
        prob_1(i,k) = PI_1 + L1_a(i,k);
        end
    end
end 

for i = 1:N
    for j = 1:57
        for k = 1 : 200
            if xij(i,j) == 1
                L0_a(i,k) = L0_a(i,k) + log (thetajc_0a(k,j));
            else
                L0_a(i,k) = L0_a(i,k) + log (1- thetajc_0a(k,j));
            end
        prob_0(i,k) = PI_0 + L0_a(i,k);
        end
    end
end 


for i = 1:N
    for k = 1:200
        if prob_1(i,k) > prob_0(i,k)
            label_a(i,k) = 1;
        else
            label_a(i,k) = 0;
        end
    end
end

count = zeros(1,200);

for k = 1:200
    for i = 1:N
        if label_a(i,k) ~= yi(i,1)
            count(1,k) = count(1,k) + 1;
        end
    end
end

for k = 1 : 200
    acc(1,k) = (N - count (1,k)) / N;
    error(1,k) = (1 - acc (1,k)) * 100;
end

%% Calculating the Accuracy for the Test data for the different values of alpha = 1 to 100

xij_test = Xtest_s;
L1_a_t = zeros (1536,200);
L0_a_t = zeros (1536,200);

for i = 1:1536
    for j = 1:57
        for k = 1 : 200
            if xij_test(i,j) == 1
                L1_a_t(i,k) = L1_a_t(i,k) + log (thetajc_1a(k,j));
            else
                L1_a_t(i,k) = L1_a_t(i,k) + log (1- thetajc_1a(k,j));
            end
        prob_1_t(i,k) = PI_1 + L1_a_t(i,k);
        end
    end
end   

for i = 1:1536
    for j = 1:57
        for k = 1 : 200
            if xij_test(i,j) == 1
                L0_a_t(i,k) = L0_a_t(i,k) + log (thetajc_0a(k,j));
            else
                L0_a_t(i,k) = L0_a_t(i,k) + log (1- thetajc_0a(k,j));
            end
        prob_0_t(i,k) = PI_0 + L0_a_t(i,k);
        end
    end
end 

for i = 1:1536
    for k = 1:200
        if prob_1_t(i,k) > prob_0_t(i,k)
            label_a_t(i,k) = 1;
        else
            label_a_t(i,k) = 0;
        end
    end
end

count_t = zeros(1,200);

for k = 1:200
    for i = 1:1536
        if label_a_t(i,k) ~= ytest(i,1)
            count_t(1,k) = count_t(1,k) + 1;
        end
    end
end

for k = 1 : 200
    acc_test(1,k) = (1536 - count_t (1,k)) / 1536;
    error_test(1,k) = (1 - acc_test (1,k)) * 100;
end

figure
plot (1:200, (1-acc)*100, 'b')
title('Alpha Vs Error Rate');
xlabel('alpha'),ylabel('Error rate (1- Accuracy)');
hold on
plot(1:200 ,(1-acc_test)*100,'r')

legend('Train','Test');
