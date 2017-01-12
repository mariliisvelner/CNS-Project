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

# Define basemodels for ensembling
h2o.randomForest.2 <- function(..., ntrees = 300, mtries = 50, max_depth = 20, nbins = 20, balance_classes = TRUE) h2o.randomForest.wrapper(..., ntrees = ntrees, mtries=mtries,max_depth=max_depth,nbins=nbins,balance_classes = balance_classes)
h2o.randomForest.4 <- function(..., ntrees = 150, nbins = 30, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 150, seed = 1, balance_classes = TRUE) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 150, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)

# Combine the ensemble
learner <- c("h2o.randomForest.2","h2o.randomForest.4", 
             "h2o.gbm.1","h2o.gbm.5")

# Define the superlearner
metalearner <- "h2o.glm.wrapper"

# Start training the ensemble, response is in 1st column
fit <- h2o.ensemble(x = predictors, 
                    y = 1, 
                    training_frame = train.hex, 
                    family = "binomial", 
                    learner = learner, 
                    metalearner = metalearner)

# Predict on test set using the trained ensemble
pred <- predict(fit, test.hex)
predictions <- as.data.frame(pred$pred)[,1]
test$predictions <- predictions

# Define the functions for evaluating mathematical performance
F1score <- function(prediction, real){F1score <- 2*(Sensitivity(prediction,real)*Precision(prediction,real))/
  (Sensitivity(prediction,real)+Precision(prediction,real))
return(F1score)
}

Sensitivity <- function(prediction,real){
  Sensitivity <- sum(prediction==1 & real==1) / sum(real==1)
  return(Sensitivity)
}

Precision <- function(prediction,real){
  Precision <- sum(prediction==1 & real==1) / sum(prediction==1)
  return(Precision)
}

# Evaluate performance on test set
modelPerformance <- cbind.data.frame(F1score = F1score(test$predictions,test$V1),
                                     Sensitivity = Sensitivity(test$predictions,test$V1), 
                                     Precision = Precision(test$predictions,test$V1),
                                     Type1Error = sum(test$predictions == 0 & test$V1 == 1),
                                     Typ2Error = sum(test$predictions == 1 & test$V1 == 0),
                                     Date = Sys.time())

modelPerformance