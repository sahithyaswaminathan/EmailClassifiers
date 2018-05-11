clc;

clear all;

load spamData.mat;

%%Preprocessing the data using Z-normalization

x_z = Xtrain;
x_tz = Xtest;

columnmean = mean(x_z,1);

columnvar = var(x_z,1);

for j = 1:57
    for i = 1:3065
        Xtrain_z(i,j) = (x_z(i,j) - columnmean(1,j))/ columnvar(1,j);
    end
end

columnmean_t = mean (x_tz,1);
columnvar_t = var (x_tz,1);

for j = 1 :57
    for i = 1:1536
        Xtest_z(i,j) = (x_tz(i,j) - columnmean_t(1,j))/ columnvar_t(1,j);
    end
end


%%Gaussian Naive Bayes for the Z normalization

z1 = zeros(3065,57);

N = 3065;

yz = ytrain;
xz = Xtrain_z;

%Creating a matrix which stores value only when ytrain == 1 'spam'%
for i = 1:N
    c = yz(i,1);
    if c == 1 
        for j = 1:57
            z1(i,j) = xz(i,j);
        end
    end
end

a1 = z1(any(z1,2),:); %Deleting all the rows with the value zero%

A1 = sum(a1);

for j = 1:57
    mu_1z(1,j) = A1 (1,j) / 1234; %Calculating Mean for class = 1%
end

for i = 1: length(a1)
    for j = 1:57
    var_0z(i,j) =(a1(i,j) - mu_1z(1,j))^2/length(a1); %Calculating Variance for class=1%
    end
end
var_1z(1,:) = sum(var_0z);
    
%Creating a matrix which stores value only when ytrain == 0 'not - spam'%
for i = 1:N
    c = yz(i,1);
    if c == 0 
        for j = 1:57
            z0(i,j) = xz(i,j);
        end
    end
end



a0 = z0(any(z0,2),:); %Deleting all the rows with the value zero%
    
A0 = sum(a0);

for j = 1:57
    mu_1z(2,j) = A0 (1,j)/1831; %Calculating Mean for class = 0%
end
for i = 1:length(a0)
    for j = 1:57
    var_00z(i,j) = (a0(i,j) - mu_1z(2,j))^2/length(a0); %Calculating Variance for class=0%
    end
end
var_1z(2,:) = sum(var_00z);


%%Calculating the class prior

pijc_1 = length(A1)/N; %prior maximum likelihood for the spam emails%
pijc_0 = length(A0)/N; %prior maximum likelihood for the non-spam email%

PI_1 = log (pijc_1);
PI_0 = log (pijc_0);

%%Calculating the probability

L1_l0z = zeros(N,2);

%Calculating the prob for 'spam' class 1 [1*3065]%
for i = 1:N
    for j = 1:57
            L1_l0z(i,1) = L1_l0z(i,1) - ((xz(i,j)-mu_1z(1,j))^2/(2*var_1z(1,j))) - log(sqrt(2*pi*var_1z(1,j))); 
    end
    probz(i,1) = PI_1 + L1_l0z(i,1);
end
       
for i = 1:N
    for j = 1:57
        L1_l0z(i,2) = L1_l0z(i,2) - ((xz(i,j)-mu_1z(2,j))^2/(2*var_1z(2,j))) - log(sqrt(2*pi*var_1z(2,j)));
    end
    probz(i,2) = PI_0 + L1_l0z(i,2);
end

for i = 1:N
if probz(i,1) > probz(i,2)
    labelz(i,1) = 1;
else
    labelz(i,1) = 0;
end
end

testcount= 0;

for i = 1:N
    if (labelz(i,1) ~= ytrain(i,1)) %Counting the mis-classified values%
        testcount = testcount + 1;
    end
end
   
accuracy = (N - testcount) / N;
error = (1-accuracy)*100;
    
    
%%Calculating the test accuracy and probability

L1_t0z = zeros(1536,2);

xtest = Xtest_z;

for i = 1:1536
    for j = 1:57
            L1_t0z(i,1) = L1_t0z(i,1) - ((xtest(i,j)-mu_1z(1,j))^2/(2*var_1z(1,j))) - log(sqrt(2*pi*var_1z(1,j)));
    end
    prob_test(i,1) = PI_1 + L1_t0z(i,1);
end
       
for i = 1:1536
    for j = 1:57
        L1_t0z(i,2) = L1_t0z(i,2) - ((xtest(i,j)-mu_1z(2,j))^2/(2*var_1z(2,j))) - log(sqrt(2*pi*var_1z(2,j)));
    end
    prob_test(i,2) = PI_0 + L1_t0z(i,2);
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
error_test = (1- accuracy_test)*100;
    