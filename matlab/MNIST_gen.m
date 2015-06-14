clear
train_labels = MNISTParser('./MNIST/train-labels-idx1-ubyte');
train_images = MNISTParser('./MNIST/train-images-idx3-ubyte');
test_labels = MNISTParser('./MNIST/t10k-labels-idx1-ubyte');
test_images = MNISTParser('./MNIST/t10k-images-idx3-ubyte');
test_item_number = length(test_labels);
train_item_number = length(train_labels);
image_scale = size(test_images,2);
test_images_unfold = reshape(test_images,test_item_number,image_scale^2)';
test_labels_unfold = full(ind2vec(test_labels'+1));
train_images_unfold = reshape(train_images,train_item_number,image_scale^2)';
train_labels_unfold = full(ind2vec(train_labels'+1));
save MNIST.mat;

colormap(gray);
axis off
axis image

%show an image of a digit in test samples
for i=1:1
    j=randi(length(test_labels),1);
    image(reshape(255-test_images(j,:,:),28,28));
    title(sprintf('%d',test_labels(j)));
    pause(1);
    image(reshape(test_images_unfold(:,j),28,28));
    title(vec2ind(test_labels_unfold(:,j))-1);
    pause(1);
end    

%show an image of a digit in train samples
for i=1:1
    j=randi(length(train_labels),1);
    image(reshape(255-train_images(j,:,:),28,28));
    title(sprintf('%d',train_labels(j)));
    pause(1);
    image(reshape(train_images_unfold(:,j),28,28));
    title(vec2ind(train_labels_unfold(:,j))-1);
    pause(1);
end
