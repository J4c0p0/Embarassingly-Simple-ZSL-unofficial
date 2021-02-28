clear; close all; clc;

% SUMMARY: PAVIS/VGM tutorial for Embarassingly Simple Zero-Shot Learning
%
%       Author : Jacopo Cavazza
%       Created : Feb 17th, 2021
%       Last Edit : Feb 28th, 2021
%
%
% Copyright (c) 2021 Jacopo Cavazza
%
% This code is a third-party implementation of the ICML 2015 paper
% entitled "An Embarassingly Simple Approach for Zero-Shot Learning" and
% authored by B. R. Paredes and P. H. Torr (from the Oxford University,
% Department of Engineering Science, Parks Road, Oxford, OX1 3PJ, UK).
% Therefore, the intellectual property of the algorithm coded here is not proprietary
% of the author of the code who downloaded the publicly available paper
% from http://proceedings.mlr.press/v37/romera-paredes15.pdf and re-coded
% by himself, exploiting publicly avaialable data (GoogleNet features 
% and splits from [Kodirov et al., Semantic Autoencoder for Zero-shot 
% Learning, CVPR 17], class embeddings by [Xian et al., Zero-Shot 
% Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly, TPAMI 18]).
%
% Permission is hereby granted, free of charge, to any person
% obtaining a copy of this software and associated documentation
% files (the "Software"), to deal in the Software without
% restriction, including without limitation the rights to use,
% copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following
% conditions:
%
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
% OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
% HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
% WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
% FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
% OTHER DEALINGS IN THE SOFTWARE.

%% Paths
  
path_2_images = '??'; %Path to Animals with Attributes 2 images;
complete_path_2_Xian_AWA2_attributes = '??'; %Path to semantic embeddings by "The Good, The Bad and The Ugly" survey by Xian et al. TPAMI 18
complete_path_2_GoogleNet_Feats = '??'; %Path to GoogleNet features by Kodirov et al.
comptete_path_2_GoogleNet_MetaData = '??'; 	%Path to splits by Kodirov et al. (note that at the time, the paper
											%simply couldn't rely on Proposed Splits by Xian et al. - which should
											%be preferred for applications. For the sake of an accurate 
											%historical reproduction of the code, I am not considering them.

%% Visualizing Images from AWA2
                               
%I want to sample 3 random class names and visualize 3 random images each
three_random_classes = dir(path_2_images);
three_random_classes(1:2) = []; %Removing parent and current folder ids

three_random_classes = three_random_classes(randperm(length(three_random_classes))); %Permute

figure('Name','Animals with Attributes Images','Color','w')
for i = 1 : 3
    tmp_image_list = dir([path_2_images '\' three_random_classes(i).name]);
    tmp_image_list(1:2) = []; %Removing parent and current folder ids
    
    tmp_image_list = tmp_image_list(randperm(length(tmp_image_list)));
    for j = 1 : 3
        subplot(3,3,(j-1)*3+i)
        imagesc(imread([tmp_image_list(j).folder '\' tmp_image_list(j).name]));
        axis image off;
        if j == 1
            title(three_random_classes(i).name,'FontSize',35)
        end
    end
end
drawnow;



%% Loading Pre-computed GoogleNet features and meta-data

load(complete_path_2_GoogleNet_Feats); %Loading features
load(comptete_path_2_GoogleNet_MetaData,'tr_loc','te_loc','Ytr','Yte','te_ind'); %Loading meta-data

%Shaping features, labels and seen/unseen splits using a better structure
Features = X';
Labels = zeros(size(X,2),1);
Labels(tr_loc) = Ytr;
Labels(te_loc) = Yte;
UnseenClassesID = te_ind';
SeenClassesID = setdiff(1:50,te_ind)';
where_seen_samples = ismember(Labels,SeenClassesID);
where_unseen_samples = ismember(Labels,UnseenClassesID);
clear tr_loc te_loc Ytr Yte te_ind X;

%% Visualizing GoogleNet features for AWA2 with t-SNE

if exist('tsne_feats_GoogleNet_AWA2.mat','file')
    load('tsne_feats_GoogleNet_AWA2.mat');
else
    tsne_2d_points = tsne(Features');
    fprintf('Computing t-SNE ..');
    save('tsne_feats_GoogleNet_AWA2.mat','tsne_2d_points');
    fprintf('done!\n')
end

figure('Name','t-SNE [van Der Maten 2013] visualization of GoogleNet features for AWA2','Color','w')
scatter(tsne_2d_points(:,1),tsne_2d_points(:,2),60,Labels,'filled');
hold on
scatter(tsne_2d_points(ismember(Labels,UnseenClassesID),1),tsne_2d_points(ismember(Labels,UnseenClassesID),2),10,'rx');
hold off
grid on
drawnow



%% Visualizing Attributes

load(complete_path_2_Xian_AWA2_attributes,'original_att','allclasses_names','att')

f2 = figure('Name','Attributes','Color','w');
axes2 = axes('Parent',f2);
imagesc(original_att)
title('Osherson''s default probability scores')
colormap parula
colorbar
axis square
set(axes2,'FontSize',30,'Layer','top',...
    'XTick',[15 31 39],...
    'XTickLabel',allclasses_names([5 31 39]),...
    'XTickLabelRotation',90,...
    'YTick',[1 25 80],...
    'YTickLabel',{'black','longneck','timid'});
annotation(f2,'rectangle',...
    [0.544010416666666 0.143008474576271 0.0114166666666652 0.786016949152542],...
    'Color',[1 0.411764705882353 0.16078431372549],...
    'LineWidth',5);
drawnow;

%% Building up the paper notation (please, check it for reference!)

z = 50;
a = 85;
S = att; clear original_att;

d = 1024;
m = sum(where_seen_samples);
X = Features(:,where_seen_samples);

Y = -ones(size(Features,2),z);
for i = 1 : m
    Y(i,Labels(i)) = 1;
end
Y = Y(where_seen_samples,:);

figure('Name','Visualizing Annotations','Color','w')
imagesc(Y(randperm(size(Y,1),200),:));
axis image
xticks([])
yticks([])
colormap gray;
ylabel('Samples','Fontsize',30);
xlabel('Classes','Fontsize',30);
drawnow;

% Defining Error function and Regularizer
L = @(P,Y) norm(P - Y,'fro').^2;
Omega = @(V,S,X,gamma,lambda,beta) [ gamma * norm(V*S,'fro')^2 + ...
    lambda * norm(X'*V,'fro')^2 + beta * norm(V,'fro')^2];

% Closed-form solution for optimization (training)
gamma_ = 1;
lambda_ = 1;
V = pinv(X*X' + gamma_ * eye(d)) * X * Y * S' * pinv(S*S' + lambda_*eye(a));

scores = transpose(Features(:,where_unseen_samples)) * V * S;
scores(:,SeenClassesID) = NaN; %ZSL inference: I kill contribution of seen classes
[~,PredictedUnseenLabels] = max(scores,[],2);
        
T1 = mean(PredictedUnseenLabels == Labels(where_unseen_samples));

fprintf('The performance of EZSL on AWA2, with gamma = lambda = 1 is %3.2f%% \n (mean top-1 classification score over unseen classes)\n',100*T1)

%% Random weights versus optimized ones

Nrep = 15; 

for n = 1 : Nrep
    Vrand = rand(size(V));
    loss_(n) = L(transpose(X)*Vrand*S,Y);
    reg_(n) = Omega(Vrand,S,X,gamma_,lambda_,gamma_ * lambda_);
end
optimal_loss = L(transpose(X)*V*S,Y);
optimal_reg = Omega(V,S,X,gamma_,lambda_,gamma_ * lambda_);

figure('Name','Loss and Regularizer Values')
hs1 = subplot(1,2,1);
plot(loss_,'o-','LineWidth',2,'MarkerFaceColor','b')
hold on
hf = plot([1 Nrep],optimal_loss*[1 1],'LineWidth',5);
hold off
xticks(1:Nrep)
xticklabels({})
xlabel('Random guesses for $V$','Interpreter','LaTeX','FontSize',30)
ylabel('$L = \| X^\top V S - Y \|_F^2$','Interpreter','LaTeX','FontSize',30);
legend(hf,'Optimal Loss','FontSize',30,'Location','Best')
set(hs1,'FontSize',30);

hs2 = subplot(1,2,2);
plot(reg_,'o-','LineWidth',2,'MarkerFaceColor','b')
hold on
hf = plot([1 Nrep],optimal_reg*[1 1],'LineWidth',5);
hold off
xticks(1:Nrep)
xticklabels({})
xlabel('Random guesses for $V$','Interpreter','LaTeX','FontSize',30)
ylabel('$\Omega = \gamma \cdot \| V S \|_F^2 + \lambda \cdot \| X^\top V \|_F^2 + \beta \cdot \|V \|_F^2$','Interpreter','LaTeX','FontSize',30);
legend(hf,'Optimal Regularizer','FontSize',30,'Location','Best')
set(hs2,'FontSize',30);

%Loss and regularized are minimized in closed form.

%% Code to evaluate performance across several hyper-parameters
fprintf('Checking performance across several hyper-parameters: [')
counter = 0;
vals = kron([1 3],10.^(-6:6));
for i = 1 : length(vals)
	for j = 1 : length(vals)
	
		counter = counter + 1;
		if mod(counter,80) == 0
			fprintf('=')
		end
	  
        gamma_ = vals(i);
        lambda_ = vals(j);
        V = pinv(X*X' + gamma_ * eye(d)) * X * Y * S' * pinv(S*S' + lambda_*eye(a));
        
        
        % Inference
        clear scores PredictedUnseenLabels;
        scores = transpose(Features(:,where_unseen_samples)) * V * S;
        scores(:,SeenClassesID) = NaN;
        [~,PredictedUnseenLabels] = max(scores,[],2);
        
        T1(i,j) = mean(PredictedUnseenLabels == Labels(where_unseen_samples));
         
    end
end
fprintf('done! \n')

%% Visualizing Results Across Different Hyper-Params


figure('Name','Performance across hyper-parameters','Color','w')
hb = bar3(0.4930 * ones(26)); %Published mean value in the EZSL paper
xticks(1:26)
yticks(1:26)
xticklabels(kron([1 3],10.^(-6:6)))
yticklabels(kron([1 3],10.^(-6:6)))
for i = 1 : length(hb)
    hb(i).FaceColor = 'r';
    hb(i).EdgeColor = 'w';
    hb(i).FaceAlpha = 0.1;
end
hold on;
hb2 = bar3(T1);
xticks(1:26)
yticks(1:26)
xticklabels(kron([1 3],10.^(-6:6)))
yticklabels(kron([1 3],10.^(-6:6)))
for i = 1 : length(hb2)
    hb2(i).FaceColor = 'g';
    hb2(i).EdgeColor = 'w';
    hb2(i).FaceAlpha = 1;
end
axis vis3d
xlabel('$\lambda$','FontSize',30,'Interpreter','LaTeX')
ylabel('$\gamma$','FontSize',30,'Interpreter','LaTeX')
hold off
legend([hb(1) hb2(1)],'Published','Obtained','FontSize',30)
