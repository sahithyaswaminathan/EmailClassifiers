%% Z- normalization Data preprocessing

clc;
clear all;

load spamData.mat;

%%Znormalization

X_z = zscore(Xtrain);
Xt_z = zscore(Xtest);

Xtrain_z = [ones(size(X_z,1),1) X_z]; %concatenating 1s on the first column to include bias term%
Xtest_z = [ones(size(Xt_z,1),1) Xt_z];

%% calculating G, H and w

a=1;

w = zeros(1,58);

lambda= [1:100];

train_count = 0;

train_count = zeros(1,100);

for b = 1:100
    w = zeros(1,58);
 for a = 1:10
            mu = 1./ (1 + exp (-(w(a,:))*transpose(Xtrain_z))); %Transpose of Xtrain_z - To perform the matrix multiplication%
            g = transpose(Xtrain_z)*transpose((mu - transpose(ytrain)))+ transpose(cat(2,w(a,1),lambda(b).* w(a,2:58))); %Exluding the regularization term for the bias value%
            s = diag(diag(transpose(mu)*(1-mu)));
            h = transpose(Xtrain_z)*s *Xtrain_z + lambda(b).*eye(58);
            w(a+1,:) = w(a,:) - transpose(h^-1 * g);
            if a==10
                theta(b,:) = w(a,:);%Final converged value of weight is stored in theta%
                disp(a);
            end
 end
 
 c(b,:) = theta(b,:)*transpose(Xtrain_z);
 e(b,:) = theta(b,:)*transpose(Xtest_z);
end

d = transpose(c);
f = transpose(e);

for b = 1:100
for i = 1:3065
    if (d(i,b) > 0)
        label(i,b) = 1;
    else
        label(i,b) = 0;
    end
end
end


for b = 1:100
    for i = 1:3065
        if (label(i,b) ~= ytrain(i,1))
            train_count(1,b) = train_count(1,b) + 1;%Counting the mis-classified value%
        end
    end
end

for b = 1:100
    train_accuracy(1,b) = (3065 - train_count(1,b)) / 3065;
    error(1,b) = (1- train_accuracy(1,b))*100;
end

 
%% Calculating the test accuracy and test count

test_count = zeros(1,100);

for b = 1:100
for i = 1:1536
    if (f(i,b) > 0)
        label_test(i,b) = 1;
    else
        label_test(i,b) = 0;
    end
end
end


for b = 1:100
    for i = 1:1536
        if (label_test(i,b) ~= ytest(i,1))
            test_count(1,b) = test_count(1,b) + 1;%Counting the mis-classified values%
        end
    end
end

for b = 1:100
    test_accuracy(1,b) = (1536 - test_count(1,b)) / 1536;
    error_test(1,b) = (1- test_accuracy(1,b))*100;
end

figure
plot (1:100, (1-train_accuracy)*100, 'b')
title('Lambda Vs Error Rate');
xlabel('lambda'),ylabel('Error rate (1- Accuracy)');
hold on
plot(1:100 ,(1-test_accuracy)*100,'r')
legend('Train','Test');
