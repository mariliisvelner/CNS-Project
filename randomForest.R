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

# Split to 75% train and 25% test
trainIdx <- base::sample(nrow(dataAll), ceiling(nrow(dataAll)*0.75))
train <- dataAll[trainIdx,]
test <- dataAll[-trainIdx,]

# Initiate a local cluster with 12G RAM and send data to cluster
h2o.init(nthreads=-1, max_mem_size="12g")

test.hex <- as.h2o(test)
train.hex <- as.h2o(train)
predictors <- colnames(dataAll[,-1])

# Train a regular RF, response in 1st column
model <- h2o.randomForest(x=predictors,
                          y = 1,
                          training_frame=train.hex,
                          validation_frame=test.hex,
                          ntrees = 100,
                          max_depth = 20,
                          nbins = 20,
                          binomial_double_trees = TRUE,
                          ignore_const_cols = FALSE,
                          stopping_metric = "AUTO")

# Evaluate the performance
model