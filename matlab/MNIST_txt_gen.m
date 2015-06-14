clear;
load MNIST;
%load MNIST_downsampled.mat
%addpath '../matLib/'

fp_tr = fopen('MNIST_train.txt','w');
fp_test = fopen('MNIST_test.txt','w');
% fp_tr = fopen(sprintf('MNIST_train_step_%d.txt',step),'w');
% fp_test = fopen(sprintf('MNIST_test_step_%d.txt',step),'w');

%% original
idx = find(max(train_images_unfold)-min(train_images_unfold));
assert(length(idx)==train_item_number);
train_images_unfold = normalization_column(train_images_unfold,'minmax')*2-1;
train_images_unfold=train_images_unfold';

idx = find(max(test_images_unfold)-min(test_images_unfold));
assert(length(idx)==test_item_number);
test_images_unfold = normalization_column(test_images_unfold,'minmax')*2-1;
test_images_unfold = test_images_unfold';
%% down sampled
% %train_img_sampled = train_img_sampled';
% idx = find(max(train_img_sampled)-min(train_img_sampled));
% %assert(length(idx)==train_item_number);
% train_img_sampled(:,idx) = normalization_column(train_img_sampled(:,idx),'minmax')*2-1;
% train_img_sampled = train_img_sampled';
% 
% idx = find(max(test_img_sampled)-min(test_img_sampled));
% %assert(length(idx)==test_item_number);
% test_img_sampled(:,idx) = normalization_column(test_img_sampled(:,idx),'minmax')*2-1;
% test_img_sampled = test_img_sampled';
%%
feature_num = size(train_images_unfold,2)
%feature_num = size(train_img_sampled,2)

%% original
fprintf(fp_tr,'%d %d %d\n',feature_num,10,train_item_number);
for i=1:train_item_number
    for j=1:feature_num
        fprintf(fp_tr,'%f ',train_images_unfold(i,j));
    end
    fprintf(fp_tr,'%d\n',train_labels(i));
end

fprintf(fp_test,'%d %d %d\n',feature_num,10,test_item_number);
for i=1:test_item_number
    for j=1:feature_num
        fprintf(fp_test,'%f ',test_images_unfold(i,j));
    end
    fprintf(fp_test,'%d\n',test_labels(i));
end

%% down sampled
% fprintf(fp_tr,'%d %d %d\n',feature_num,10,train_item_number);
% for i=1:train_item_number
%     for j=1:feature_num
%         fprintf(fp_tr,'%f ',train_img_sampled(i,j));
%     end
%     fprintf(fp_tr,'%d\n',train_labels(i));
% end
% 
% fprintf(fp_test,'%d %d %d\n',feature_num,10,test_item_number);
% for i=1:test_item_number
%     for j=1:feature_num
%         fprintf(fp_test,'%f ',test_img_sampled(i,j));
%     end
%     fprintf(fp_test,'%d\n',test_labels(i));
% end

%%
fclose(fp_tr);
fclose(fp_test);