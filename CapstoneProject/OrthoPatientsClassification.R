# This script is the Capstone Own Project for HarvardX Data Science Professional Certificate.
# Dataset chosen to study classification of Orthopedic Patients based on Biomechanical Features.
# Data has been downloaded from Kaggle database.In this project KNN (Nearest Neighbour Algorithm)
# will be used for patients classification.


# Why have i chose the data
# I have  chosen this dataset because:
# It is freely avaliable online on Kaggle.
# It is 'medium' sized. Not small or too big to be procecssed on my personal computer.
# There are a reasonable number of predictor columns, and easy to understand.


# Step1: Load Library Packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gmodels)) install.packages("gmodels", repos = "http://cran.us.r-project.org")

library(gmodels)
library(tidyverse)
library(caret)
library(data.table)
library(corrplot)
library(ggplot2)
library(class)

# Step2: Load Data
# Data file is stored in github for easy
# Getting data file from https://github.com/cmudgal/DataScienceHarvardX/tree/master/CapstoneProject/OrthoPatientData

u="https://raw.githubusercontent.com/cmudgal/DataScienceHarvardX/master/CapstoneProject/OrthoPatientData/column_2C_weka.csv"
ortho_data <- read.csv(u)#"Column_2C_weka.csv")

# Peek at first 10 rows of data
head(ortho_data, n=10)

# Peek at last 10 rows of data
tail(ortho_data, n=10)

# Get number or rows and columns in data set
# Data set has 310 rows and 7 columns
dim(ortho_data)

# Get summary of the data
summary(ortho_data)


# Gives the structure of data info
# All non null data set with 7 variables
# 6 are numeric features
# class is the factor with levels "Abnormal" and "Normal"
str(ortho_data)

# Explore Data for Orthopedic Patients
plot(ortho_data$class, freq=TRUE, col="red", border="white", 
     main="Distribution of patients", xlab="Class", ylab="Count")

# There are 210 patients in Abnormal and 100 in Normal category 
# Divison of 'class' attribute of the patients
table(ortho_data$class) 

# Percentual division of patients using the  `class` attribute
round(prop.table(table(ortho_data$class)) * 100, digits = 1)


# Correlation plot
#From the figure, as it seems Normal class values are smaller than Abnormal values; therefore i 
#need to narrow with selecting some correlated features. Lets look at correlation matrix of all features:
com = ortho_data[,1:6]
cc = cor(com, method = "spearman")
corrplot(cc, tl.col = "black", order = "hclust", hclust.method = "average", addrect = 4, tl.cex = 0.7)

# There are highly positive correlations between Pelvic Incidence and Sacral Slope
# , also, between Pelvic Incidence and Lumbar Lordosis Angle as can be seen
# by scatter plots below.

# Scatter plot for pelvic_radius and sacral_slope for distribution of patients
ggplot(ortho_data, aes(x = pelvic_radius, y = sacral_slope)) +
  geom_point(aes(color = factor(class)))

# Scatter plot for pelvic_incidence and sacral_slope for distribution of patients

ggplot(ortho_data, aes(x = pelvic_incidence, y = sacral_slope)) +
  geom_point(aes(color = factor(class)))

# Scatter plot for pelvic_incidence and Lumbar Lordosis Angle for distribution of patients

ggplot(ortho_data, aes(x = pelvic_incidence, y = lumbar_lordosis_angle)) +
  geom_point(aes(color = factor(class)))

#heat map
palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = cc, col = palette, symm = TRUE)

# Prepare Data
# The Biomechanical Orthopedic data set will be used for classification, which is an example of predictive modeling. 
# The last attribute of the data set, class, will be the target variable or the variable that I want to predict. 
# 1) Check of null values 2) normalize data 3) Split data in test and train data sets

# check if isna

data_na<-apply(ortho_data, 2, function(x) any(is.na(x)))
data_na
# There is no na in the dataset

# Normalize data: Looking at the summary output it is seen that all the features are not in consistent range.
# Look at the minimum and maximum values of all the (numerical) attributes. If one attribute has a wide range of values,
# need to normalize the dataset, because this means that the distance will be dominated by this feature. In the
# current dataset it is degree_spondylolosthesis that has wide range from -11.058 to 418.543

# Build normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

# Apply normalize function to the orthopedic data set

ortho_data.norm<-as.data.frame(lapply(ortho_data[,c(1,2,3,4,5,6)], normalize))

# Summary for normalized data
summary(ortho_data.norm)

# Split Data into Training and Test Sets
# In order to assess the performance of the mode data set is divided into two parts: a training set and a test set.

# The first is used to train the system, while the second is used to evaluate the learned or trained system. 
# 70% of the original data set is as the training set, while the 30% that remains will compose the test set.

# To make training and test sets,  set a seed. This is a number of R’s random number generator. 
# The major advantage of setting a seed is that it gives same sequence of random numbers.
set.seed(123)
ortho_data.ind <- sample(1:nrow(ortho_data.norm),size=nrow(ortho_data.norm)*0.7,replace = FALSE) #random selection of 70% data

ortho_data.train <- ortho_data.norm[ortho_data.ind,] # 70% training data

# Inspect training set
head(ortho_data.train)

ortho_data.test <- ortho_data.norm[-ortho_data.ind,] # remaining 30% test data


# Inspect test set
head(ortho_data.test)

# Compose `class` training labels

ortho_data.trainLabels <- ortho_data[ortho_data.ind,7]

# Inspect result
print(ortho_data.trainLabels)

# Compose `class` test labels
ortho_data.testLabels <- ortho_data[-ortho_data.ind, 7]

# Inspect result
print(ortho_data.testLabels)

# KNN Algorithm
# Build Classifier to find the k nearest neighbour for the training set
# using the knn() function, which uses the Euclidian distance measure to find the k-nearest 
# neighbours to the new instance.

# To find the k parameter for the knn function
nr<-NROW(ortho_data.trainLabels) 
# sqrt of 217 is 14.7
nr
sqrt(nr)



dim(ortho_data.trainLabels)
dim(ortho_data.train)
ortho_data.knn.15 <- knn(train=ortho_data.train, test=ortho_data.test, cl=ortho_data.trainLabels, k=14)
ortho_data.knn.16 <- knn(train=ortho_data.train, test=ortho_data.test, cl=ortho_data.trainLabels, k=15)

# Inspect
ortho_data.knn.15

# ortho_data.knn.15 stores the knn() function that takes as arguments 
# the training set, the test set, the train labels and the amount of 
# neighbours to find with this algorithm. The result of this function 
# is a factor vector with the predicted classes for each row of the test data.

# Note that the test labels will be used to see if the model is good at prediction.

# Model Evaluation
# An essential next step in machine learning is the evaluation 
# of the model’s performance. Analyze the degree of correctness of the model’s predictions.

# Put `ortho_data.testLabels` in a data frame
ortho_dataTestLabels <- data.frame(ortho_data.testLabels)

# Merge `ortho_data.knn.15` and `ortho_data.testLabels` 
merge <- data.frame(ortho_data.knn.16, ortho_data.testLabels)



# Inspect `merge` 
merge


CrossTable(x = ortho_data.testLabels, y = ortho_data.knn.16, prop.chisq=FALSE)



##create confusion matrix
tab <- table(ortho_data.knn.16, ortho_data.testLabels)

##this function divides the correct predictions by total number of predictions that tell us how accurate the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

#Calculate the proportion of correct classification for k = 15,16
ACC.16 <- 100 * sum( ortho_data.testLabels == ortho_data.knn.15)/NROW( ortho_data.testLabels)
ACC.17 <- 100 * sum( ortho_data.testLabels == ortho_data.knn.16)/NROW( ortho_data.testLabels)
ACC.16
ACC.17



confusionMatrix(table(ortho_data.knn.16 , ortho_data.testLabels))
#Confusion Matrix and Statistics

##############################################
####################Optimization

i=1
k.optm=1
for (i in 1:28){
  knn.mod <- knn(train=ortho_data.train, test=ortho_data.test, cl=ortho_data.trainLabels, k=i)
  k.optm[i] <- 100 * sum( ortho_data.testLabels == knn.mod)/NROW( ortho_data.testLabels)
  k=i
  cat(k,'=',k.optm[i],'
')
}

#Accuracy plot
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

################### Using Caret Package
# Create index to split based on labels  
index <- createDataPartition(ortho_data$class, p=0.7, list=FALSE)

# Subset training set with index
ortho.training <- ortho_data[index,]

# Subset test set with index
ortho.test <- ortho_data[-index,]

# Overview of algos supported by caret
#names(getModelInfo())

# Train a model
model_knn <- train(ortho.training[, 1:6], ortho.training[, 7], method='knn')

# Predict the labels of the test set
predictions<-predict(object=model_knn,ortho.test[,1:6])

# Evaluate the predictions
table(predictions)

# Confusion matrix 
confusionMatrix(predictions,ortho.test[,7])

#### SVM Linear (Supported Vector Machine)

# Train a model
model_svm <- train(ortho.training[, 1:6], ortho.training[, 7],method='svmLinear',trControl=trainControl(method='cv',number=10),preProcess = c("center", "scale"))

# Predict the labels of the test set
predictions<-predict(object=model_svm,ortho.test[,1:6])

# Evaluate the predictions
table(predictions)

# Confusion matrix 
confusionMatrix(predictions,ortho.test[,7])


#### random forest

# Train a model random forest
model_cnn <- train(ortho.training[, 1:6], ortho.training[, 7], method='rf')

# Predict the labels of the test set
predictions<-predict(object=model_cnn,ortho.test[,1:6])

# Evaluate the predictions
table(predictions)


# Confusion matrix 
confusionMatrix(predictions,ortho.test[,7])

