% Delta Rule
% This will create some random data and assign it to a class. It will then
% train a neural network with the delta rule with a static learning rate
% and graph the error.
% Tyler Rose and Seth Dippold
clear all; close all;

% Create some random data around x1 + 2*x2 - 2 and classify it as > 0 is 1
% and <=0 is 0
N = 100;
data = gen_sigmoid_classes(N);
x = [data(:,1),data(:,2)];
y = data(:,3);

% initialize the weight vector
w = rand(1,2+1);

% Batch Fashion
maxIterations = 100;
iterations = 0;
eta = .01; % this is the learning rate
tic;
while (iterations < maxIterations)
    iterations = iterations + 1;
    for i=1:N
        out(i) = w(3) + w(2).*(x(i,2) + x(i,2).^2) + w(1).*(x(i,1) + x(i,1).^2);
        deltaW = eta*(y(i) - out(i))*[x(i,:),1];
        w = w + deltaW;
        err(i) = (y(i) - out(i))^2;
    end
    E(iterations) = sum(err)/N;
end
toc
min(E)
% Plot the Error per iteration
plot(linspace(1,maxIterations,maxIterations),E);
title('Error per Iteration');
ylabel('Error');
xlabel('Iteration');

% Incremental Fashion
% maxIterations = 100;
% iterations = 0;
% numUpdates = 0;
% eta = .01; % this is the learning rate
% tic;
% for i=1:N
%     out(i) = sum(w.*[x(i,:),1]);
%     deltaW = eta*(y(i) - out(i))*[x(i,:),1];
%     w = w + deltaW;
%     err(i) = (y(i) - out(i))^2;
%     E(i) = sum(err)/i;
% end
% toc
% min(E)
% return
% % Plot the Error per iteration
% plot(linspace(1,maxIterations,maxIterations),E);
% title('Error per Iteration');
% ylabel('Error');
% xlabel('Iteration');
