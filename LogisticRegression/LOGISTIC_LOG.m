%%Loading the initial data

clc;

clear all;

load spamData.mat;

%% Log data preprocessing

N = 3065;

yl = ytrain;
Xtrain_l = log(Xtrain+0.1);

xlog = [ones(size(Xtrain_l,1),1) Xtrain_l]; %concatenating 1s on the first column to include bias term%

Xtest_l = log(Xtest+0.1);

xtest = [ones(size(Xtest_l,1),1) Xtest_l];


%% calculating G, H and w

a=1;

w = zeros(1,58);

lambda= [1:100];

train_count = 0;

train_count = zeros(1,100);

for b = 1:100
    w = zeros(1,58);
 for a = 1:10
            mu = 1./ (1 + exp (-(w(a,:))*transpose(xlog))); %Transpose of xlog - To perform the matrix multiplication%
            g = transpose(xlog)*transpose((mu - transpose(ytrain)))+ transpose(cat(2,w(a,1),lambda(b).* w(a,2:58))); %The regularization term is excluded for the bias value%
            s = diag(diag(transpose(mu)*(1-mu)));
            h = transpose(xlog)*s *xlog + lambda(b).*eye(58);
            w(a+1,:) = w(a,:) - transpose(h^-1 * g);
            if a==10
                theta(b,:) = w(a,:);%Final converged weights are stored in theta for each lambda value%
                disp(a);
            end
 end
 
 c(b,:) = theta(b,:)*transpose(xlog);
 e(b,:) = theta(b,:)*transpose(xtest);
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
        if (label(i,b) ~= ytrain(i,1)) %Counting the mis-classified value%
            train_count(1,b) = train_count(1,b) + 1;
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
            test_count(1,b) = test_count(1,b) + 1; %Counting the mis-classified value%
        end
    end
end

for b = 1:100
    test_accuracy(1,b) = (1536 - test_count(1,b)) / 1536;
    error_test(1,b) = (1- test_accuracy(1,b))*100;
end

%%Plot
figure
plot (1:100, (1-train_accuracy)*100, 'b')
title('Lambda Vs Error Rate');
xlabel('lambda'),ylabel('Error rate (1- Accuracy)');
hold on
plot(1:100 ,(1-test_accuracy)*100,'r')
legend('Train','Test');
