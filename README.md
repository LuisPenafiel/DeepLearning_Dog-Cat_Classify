# Deep Learning dog&cat_classifier
## Instructions
### Description of the problem
We want to train a classifier using deep learning to recognize and classify images of dogs and cats.

#### setup 0: Pip & Requirements, Imports, Initalization

#### Step 1: Loading the dataset
The dataset is located in Kaggle and you will need to access it to download it. You can find the competition here (or by copying and pasting the following link in your browser: https://www.kaggle.com/c/dogs-vs-cats/data)

Download the dataset folder and unzip the files. You will now have a folder called train containing 25,000 image files (.jpg format) of dogs and cats. The pictures are labeled by their file name, with the word dog or cat.

#### Step 2: Visualize the input information
The first step when faced with a picture classification problem is to get as much information as possible through the pictures. Therefore, load and print the first nine pictures of dogs in a single figure. Repeat the same for cats. You can see that the pictures are in color and have different shapes and sizes.

#### Step 3: Build an ANN
Any classifier that fits this problem will have to be robust because some images show the cat or dog in a corner, or perhaps 2 cats or dogs in the same picture. If you have been able to research some of the winner implementations of other competitions also related to images, you will see that VGG16 is a CNN architecture used to win the Kaggle ILSVR (Imagenet) competition in 2014. It is considered one of the best performing vision model architectures to date.

#### Step 4: Optimize the above model

Load the best model from the above and use the test set to make predictions.

#### Step 5: Save the model
Store the model in the corresponding folder.