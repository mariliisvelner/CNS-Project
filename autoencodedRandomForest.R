# Load necessary libraries
library(h2o)
library(h2oEnsemble)
library(dplyr)
library(data.table)

# Load the precalculated data from csv-s
setwd("~/Downloads")
nonseizure <- as.data.frame(fread("dog1.inter.csv",sep=",",header=FALSE))
preseizure <- as.data.frame(fread("dog1.ictal.csv",sep=",",header=FALSE))

# Bind the dataframes and set the output to categorical feature
dataAll <- rbind.data.frame(nonseizure,preseizure)
dataAll[,1] <- as.factor(dataAll[,1])

# Set seed for reproducible results and take a random 30% sample of data
set.seed(234822)
sampleIdx <- base::sample(nrow(dataAll), ceiling(nrow(dataAll)*0.3))
dataAll <- dataAll[sampleIdx,]

# Initiate a local cluster with 12G RAM
h2o.init(nthreads=-1, max_mem_size="12g") 

# Define predictors/features and send the training file to cluster
predictors <- colnames(dataAll[,-1])
regular_train.hex <- as.h2o(dataAll)

# Train a deep autoencoder for dim reduction
ae_model <- h2o.deeplearning(x=predictors,
                             training_frame=regular_train.hex,
                             activation="Tanh",
                             autoencoder=T,
                             hidden=c(250, 200, 150, 100, 50),
                             ignore_const_cols=F,
                             epochs=20)

# Get the 4th layer with 100 neurons/features
features_ae <- as.data.frame(h2o.deepfeatures(ae_model, regular_train.hex, layer=4))
features_ae$label <- as.factor(dataAll$V1)

# Remove intermediate datasets
rm(regular_train.hex, sampleIdx, nonseizure, preseizure)

# Split to 75% train and 25% test
trainIdx <- base::sample(nrow(features_ae), ceiling(nrow(features_ae)*0.75))
train <- dataAll[trainIdx,]
test <- dataAll[-trainIdx,]
predictors <- colnames(dataAll[,-101])

# Send data to cluster and train the random forest
train.hex <- as.h2o(train)
test.hex <- as.h2o(test)

# Response column is 101st or 102nd
model <- h2o.randomForest(x=predictors,
                          y = 101,
                          training_frame=train.hex,
                          validation_frame=test.hex,
                          ntrees = 100,
                          max_depth = 20,
                          nbins = 20,
                          binomial_double_trees = TRUE,
                          ignore_const_cols = FALSE,
                          stopping_metric = "AUTO")

# Evaluate the results
model