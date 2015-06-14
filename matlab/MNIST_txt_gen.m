clear;
load MNIST;

fp_tr = fopen('MNIST_train.txt','w');
fp_test = fopen('MNIST_test.txt','w');

%% original
idx = find(max(train_images_unfold)-min(train_images_unfold));
assert(length(idx)==train_item_number);
train_images_unfold = normalization_column(train_images_unfold,'minmax')*2-1;
train_images_unfold=train_images_unfold';

idx = find(max(test_images_unfold)-min(test_images_unfold));
assert(length(idx)==test_item_number);
test_images_unfold = normalization_column(test_images_unfold,'minmax')*2-1;
test_images_unfold = test_images_unfold';


%%
feature_num = size(train_images_unfold,2)


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

%%
fclose(fp_tr);
fclose(fp_test);
