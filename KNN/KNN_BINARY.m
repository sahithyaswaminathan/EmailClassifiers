clc;

clear all;

load spamData.mat;

%%KNearest Neighbour for the Binarized values

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


%Finding the Eucledian distance%

dis = pdist2(Xtest_s,Xtrain_s,'Hamming');

[K, Index] = sort(dis,2); %Index of the specific sorted element%

k = [1:9,10:5:100];

for i = 1:28
    Knn = K_NN(ytrain,Index, k(i));
    A(:,i) = Knn;
end

   
%%Finding the Accuracy for all the K values
test_count = zeros(1,28);

for j = 1:28
    for i = 1:1536
        if A(i,j) ~= ytest(i,1)
            test_count(1,j) = test_count(1,j)+1 %Counting the mis-classified values%
        end
    end
end
   
test_accuracy = zeros(1,28);
for j = 1:28
    test_accuracy(1,j) = (1536 - test_count(1,j)) / 1536;
    test_error(1,j) = (1- test_accuracy(1,j))*100;
end

%% Finding the train accuracy for all values of K

dis1 = pdist2(Xtrain_s,Xtrain_s,'Euclidean');

[K1, Index1] = sort(dis1,2); %Index of the specific sorted element%

k = [1:9,10:5:100];

for i = 1:28
    Knn = K_NN(ytrain,Index1, k(i));
    A1(:,i) = Knn;
end

   
%%Finding the Accuracy for all the K values
train_count = zeros(1,28);

for j = 1:28
    for i = 1:3065
        if A1(i,j) ~= ytrain(i,1)
            train_count(1,j) = train_count(1,j)+1 %Counting the number of mis-classified value%
        end
    end
end
   
train_accuracy = zeros(1,28);
for j = 1:28
    train_accuracy(1,j) = (3065 - train_count(1,j)) / 3065;
    error(1,j) = (1- train_accuracy(1,j))*100;
end

%%Plot

figure
plot (k, (1-test_accuracy)*100, 'b')
title('K Vs Error Rate');
xlabel('K'),ylabel('Error rate (1- Accuracy)');
hold on
plot(k ,(1-train_accuracy)*100,'r')
legend('Test data', 'Train data');
