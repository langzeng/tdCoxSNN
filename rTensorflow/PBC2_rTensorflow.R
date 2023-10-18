rm(list=ls())
gc()

setwd("directory to tdCoxSNN")
source("funcs_util/funcs.R")
source("rTensorflow/loss_tdCoxSNN_rTensorflow.R")

library(tensorflow)
tf$constant("Hello Tensorflow!")
tf$keras$backend$set_floatx('float32')
library(keras)

# Load data
load("Data/R/pbc2long_train.Rdata")
load("Data/R/pbc2long_test.Rdata")

landmarkmonth <- 3
# Select subjects survived beyond landmarkmonth
index_train_landmark_id <- (pbc2long_train$tstop_final/30 > landmarkmonth)
index_test_landmark_id <- (pbc2long_test$tstop_final/30 > landmarkmonth)
# Select visits survived beyond landmarkmonth
index_test_landmark_visit <- (pbc2long_test$tstart/30 > landmarkmonth)

pbc2_train <- pbc2long_train %>% filter(index_train_landmark_id)

# For each subject in test dataset, use their first visit after landmark time to predict
# The rest visits are treated as truth for validation
pbc2_test <- pbc2long_test %>% filter(index_test_landmark_id & index_test_landmark_visit) %>% 
  arrange(id,tstart) %>% 
  group_by(id) %>% 
  filter(row_number()==1) %>% 
  ungroup()

feature = c(paste0('xt',1:20),paste0('x',1:7))
feature_tobe_scaled = c(paste0('xt',1:7),'x7')

x_train <- pbc2_train %>% select(all_of(feature))
# scale predictors
x_train_scale1 <- colMeans(x_train[,feature_tobe_scaled])
x_train_scale2 <- apply(x_train[,feature_tobe_scaled],2,sd)
x_scale <- function(x,mean = x_train_scale1, sd = x_train_scale2){
  x = unlist(x)
  return((x-mean)/sd)
}

x_train[,feature_tobe_scaled] <- apply(x_train[,feature_tobe_scaled],1,x_scale) %>% 
  base::t()
x_train <- x_train %>% tf$cast(.,tf$float32)

x_test <- pbc2_test %>% select(all_of(feature))
x_test[,feature_tobe_scaled] <- apply(x_test[,feature_tobe_scaled],1,x_scale) %>% 
  base::t()
x_test <- x_test %>% tf$cast(.,tf$float32)

y_train = pbc2_train %>% select(tstart,tstop,event) %>% tf$cast(.,tf$float32)
y_test = pbc2_test %>% select(tstart,tstop,event) %>% tf$cast(.,tf$float32)


### set up DNN parameters ###
num_nodes <- 30             # number of nodes per hidden layer
num_lr <- 0.01              # learning rate
num_dr <- 0.2               # dropout rate
num_epoch <- 20             # number of epoches for optimization
batch_size <- 50            # number of batch size for optimization

tf$keras$backend$clear_session()
model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(units = num_nodes, activation = 'selu') %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = num_dr) %>% 
  layer_dense(units = 1, activation = 'linear',use_bias = FALSE)

summary(model)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = num_lr),
  loss = loss_tdCoxSNN_rTensorflow,
  metrics = NULL)

model %>% fit(x_train, y_train, epochs = num_epoch, batch_size = batch_size)


# Calculate the risk score of training samples and test samples
rs_train <- model %>% predict(x_train)
rs_test <- model %>% predict(x_test)
test_risk_score <- pbc2_test %>% select(id,time) %>% mutate(rs = rs_test)

# baseline hazard function
baseline_h <- baseline_hazard(cbind(y_train,rs_train))

# survival probability
S <- survprob(time_of_interest = c(1,30,60,180,365), # in days
              haz = baseline_h, 
              test_risk_score = test_risk_score)
