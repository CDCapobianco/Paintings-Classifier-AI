# Paintings-Classifier-AI
Deep Learning Classifier based on a Convolutional Neural Network which recognizes 5 types of paintings

# The Dataset
The cleaned dataset contains roughly 8k usable pics in .jpg and .png format.
These are divided in 5 classes:

Drawings:

![3424_mainfoto_05](https://user-images.githubusercontent.com/100691347/164914001-d8e61f63-faf5-415d-a02b-b1fe9c7872b2.jpg)


Engravings:

![222](https://user-images.githubusercontent.com/100691347/164914004-58c29e1c-129f-4d44-b8f4-ae012cfb2052.jpg)


Iconography:

![247](https://user-images.githubusercontent.com/100691347/164914019-96641dd1-d205-4910-9774-7f639b96de38.jpg)

Paintings:

![0003](https://user-images.githubusercontent.com/100691347/164914038-a9cb1a1b-a7e7-42be-9556-fa1eef1aa757.jpg)


Sculptures:

![42](https://user-images.githubusercontent.com/100691347/164914041-8a1d82dd-6005-4035-81ce-00551ee31f94.jpg)


# The Model
The definitive model architecture consists of 3 Convolutional layers with max pooling, a Dense final layer and Dropout with p=0.4 after every hidden layer (Convolutional and Dense both).


<img width="439" alt="Immagine 2022-04-23 180222" src="https://user-images.githubusercontent.com/100691347/164914235-14fa4b5d-5daf-4817-881d-db63ae83d589.png">


# Performance
The model has been tested with a train set and a validation set (split of 20%) for 35 epochs, with an initial learning rate of 0.1 and a ReduceOnPlateau Scheduler to find the minima in the loss landscape.
The final results are 85% accuracy on test set.

# Final Considerations
Looking at the confusion matrix, it is clear that drawings and engravings are the most mispredicted classes for the model.

![download](https://user-images.githubusercontent.com/100691347/164914472-1f914dab-d260-4412-a9da-1404606f2b6f.png)

Engravings get confused with drawings and vice versa, and drawings get also confused with sculptures.
While the first two cases are quite understandable, since many engravings images are quite similar to drawings, it surprises that sculptures get confused with drawings.
Although if we take a look at some examples, we can see that there are sculptures with pencil-like features, like this one:

![193 18 59 45](https://user-images.githubusercontent.com/100691347/164914963-94166927-f345-41a8-978b-17a511efe5f9.jpg)

and others (due to bad lighting and/or resolution) with few details and almost solid colors like this:

![266](https://user-images.githubusercontent.com/100691347/164915126-41976f7d-a086-48ba-baa1-6e67e90d93c8.jpg)

The accuracy score can surely get improved starting by these considerations, using data augumentation or slightly deepening the model. Also, the dataset is probably too small to properly train a CNN from scratch, so probably transfer learning (+ fine tuning) could probably give a better result.

# UPDATE:
Trained a custom model made upon MobileNetV2 (roughly 2 millions paramater, 1/3 of the CNN made from scratch) on the same dataset.
It performed a 94% accuracy on evaluation, against 85% accuracy on the old model.

