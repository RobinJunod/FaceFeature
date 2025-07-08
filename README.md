# Facial Feature Detection from Vanilla Dataset

This is a personal project. the idea is to use a simple dataset in order to use facial feature detection on more complex data. The idea is to use a kaggle CSV file with grey images of size 96x96. to have something capable of detecting facial feature on webcam. this is the base of algorithm used for face filter like on snapchat. 

## The Dataset 
It consist of some faces images, very well centered. 
### Two kind of labelling
- 30 labels (15 facial keypoints). 
- <30 labels 

The solution to ceal with that is here to customize the loss function based on the number of available features

### Vanilla images

The images are very simple which means it is very easy to have a performent algorithm that works on the dataset. however to generalized it it is a other thing. in order to make it robust to all kind of images, the 