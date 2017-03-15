# AZAVEA DATA ANALYST DATA TEST -------------------------------------------

# Simon Kassel
# March 2017

# This script includes all of the code I used to answer question two of the
# evaluation (the Applied Question). I have split it up into sections 
# corresponding to the four different components of question two. I have also
# included some code in the introductory section that sets up my analysis.

# I have also compiled this code along with extensive annotation, console
# outputs, and plots in an .Rmd file that can be found in the GitHub 
# repository. Please see that document for answers to question 1.

# There are no additional instructions to run the script.

# packages
for (p in c("ggplot2", "caret", "tidyr", "C50", 
            "gmodels", "class", "randomForest", "plyr")) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p)
  suppressMessages(library(p, character.only = TRUE))
} 

# global options
options(scipen = "999")
options(stringsAsFactors = FALSE)

# data
cars <- read.csv("https://raw.githubusercontent.com/eneedham/data-analyst-data-test/master/Cars_mileage.csv")

# define standard theme for all plots
plot_theme <- function(base_size = 12) {
  theme_minimal() %+replace%
    theme(
     axis.title = element_text(hjust = 1, face = 'italic'),
     strip.text = element_text(face = 'italic'),
     plot.title = element_text(hjust = 0.5, face = 'bold', vjust = 0.25),
     legend.title = element_text(face = 'italic')
    )
}

# 2a ----------------------------------------------------------------------
# TASK: Create a binary variable that represents whether the car's mpg 
# is above or below its median. Above the median should be represented as 1. 
# Name this variable mpg_binary.

# create binary variable:
cars$mpg_binary <- ifelse(cars$mpg > median(cars$mpg), 1, 0) %>% as.factor()

# there should be roughly the same number of 1's and 0's
table(cars$mpg_binary)

# it looks like there are 9 cars that have exactly the median gas mileage,
# explaining the discrepancy
nrow(cars[which(cars$mpg == median(cars$mpg)), ])

# 2b ----------------------------------------------------------------------
# TASK: Which of the other variables seem most likely to be useful in 
# predicting whether a car's mpg is above or below its median? Describe your 
# findings and submit visual representations of the relationship between 
# mpg_binary and other variables.

# transform data to long form
cars_long <- cars %>% gather(var, val, -c(mpg_binary, name, mpg))
cars_long$val <- cars_long$val %>% as.numeric()

# split the data in two based on whether the iv is continuous or discrete 
cars_cont <- cars_long[!(cars_long$var %in% c("origin", "year")), ]
cars_disc <- cars_long[cars_long$var %in% c("origin", "year"), ]

# visualize the relationships between mpg_binary and disc. predictors 
# using normalized bar charts
ggplot(cars_disc, aes(factor(val), fill = mpg_binary)) + 
  geom_bar(position = "fill", color = "white") + 
  facet_wrap(~var, scales = "free") + 
  xlab("value") + ylab("proportion") +
  ggtitle("Discrete predictor variables") + plot_theme()

# visualize relationships with continuous variables using violin plots
ggplot(cars_cont, aes(mpg_binary, val, fill = mpg_binary)) + 
  geom_violin(color = "white") + 
  facet_wrap(~var, scales = "free") + 
  ggtitle("Continuous predictor variables") + plot_theme()

# boxplots are similar but give clear indications of median and quartiles
ggplot(cars_cont, aes(mpg_binary, val, color = mpg_binary)) + 
  geom_boxplot() + 
  facet_wrap(~var, scales = "free") + 
  ggtitle("Continuous predictor variables") + plot_theme()

# test for statistically significant differences of two variables using t.tests
#   displacement
t.test(cars$displacement ~ cars$mpg_binary)
#   acceleration
t.test(cars$acceleration ~ cars$mpg_binary)

# 2c ----------------------------------------------------------------------
# TASK: Split the data into a training set and a test set

# before splitting the data into training and test sets I need to do some
# quick cleaning, changing the type for a few variables and removing the
# name and mpg variables which would overfit the model
cars$horsepower <- cars$horsepower %>% as.numeric()
cars$year <- cars$year %>% as.factor()
cars$origin <- cars$origin %>% as.factor()

dat <- cars[ , !names(cars) %in% c("mpg", "name")]
dat <- na.omit(dat)

# partition data
set.seed(123)
in_train <- createDataPartition(dat$mpg_binary, p = .75, list = FALSE)
train_set <- dat[in_train, ]
test_set <- dat[-in_train, ]

# how many observations are in the training and test sets
paste("training set observations:", nrow(train_set), sep = " ")
paste("test set observations:", nrow(test_set), sep = " ")

# 2d ----------------------------------------------------------------------
# TASK: Perform two of the following in order to predict mpg_binary:

# I will use two different simple machine learning methods (K-Nearest Neighbors
# and Decision Trees) to predict whether each will have high or low gas mileage.
# I have broken this task up into three different sub-sections. First I
# manually build both models using default parameters. I train each on the
# training data and predict for the test set. Next, I use caret's model tuning
# functions to tune each model's parameters and see if I can improve predictive
# accuracy. Finally, I use a 10-fold cross-validation on the whole dataset
# in order to evaluate the performance of both models.

#### Secition 1 ####

# K-NEAREST NEIGHBORS

#   train model and predict
test_set$knn_prediction <- knn(train_set, test_set[,c(1:8)], train_set$mpg_binary)
test_set$knn_accuracy <- ifelse(test_set$knn_prediction == test_set$mpg_binary, 'correct', 'incorrect')

#   look at accuracy rate
table_knn <- prop.table(table(test_set$knn_accuracy)) %>% print() %>% unname()
#   and confusion matrix
CrossTable(test_set$mpg_binary, test_set$knn_prediction)

# DECISION TREE
#   train a model
dt_model <- C5.0(train_set[-8], train_set$mpg_binary)
summary(dt_model)

#   use model to predict for mpg_binary
test_set$dt_prediction <- predict(dt_model, test_set)
test_set$dt_accuracy <- ifelse(test_set$dt_prediction == test_set$mpg_binary, 'correct', 'incorrect')

#   accuracy rate
table_dt <- prop.table(table(test_set$dt_accuracy)) %>% print() %>% unname()
#   confusion matrix
CrossTable(test_set$mpg_binary, test_set$dt_prediction)

# create a df of accuracy rates for both models
results_table <- rbind(table_knn, table_dt) %>% data.frame() 
names(results_table) <- c("correct", "incorrect")
results_table$model <- c("knn", "dt")

# helper function that neatly transforms the dataset from wide to long-form
#   it takes:
#     a set of prediction outcome variables from the test set df
#   and returns:
#     a long-form data frame with the input variables as measure vars
to_long_format <- function(accuracy_variables) {
  test_long <- test_set[ , accuracy_variables] %>% gather(model, outcome)
  test_long$model <- gsub("_accuracy", "", test_long$model)
  test_long$outcome <- factor(test_long$outcome, levels = c('incorrect', 'correct'))
  return(test_long)
}

# another helper function that plots the resulting long-form dataset
# and visualizes accuracy for each model
#   it takes:
#     no parameters (the name of the data frame is hard-coded)
#   and returns:
#     a stacked bar chart showing accuracy rate for each model
accuracy_plot <- function() {
  ggplot(test_long, aes(x = model, fill = outcome)) + 
    geom_bar(stat = "count", position = "fill", alpha = 0.5) + 
    ylab("accuracy rate") + ggtitle("Comparative model accuracy rates") +
    geom_label(data = results_table, aes(x = model, y = correct, label = round(correct, 3)), 
               fill = "white") +
    plot_theme()
}

# plot accuracy for first two models
test_long <- to_long_format(c("knn_accuracy", "dt_accuracy"))
accuracy_plot()

#### Secition 2 ####

# helper function to tune the model using the train command from 
# caret, this function uses a repeated cross-validation control object
# to determine which combination of model parameters results in the 
# best accuracy rate for an out-of-sample test set. 
#   it takes:
#     a model type (string) matching the specific identifier for each algorithm
#   and returns
#     a model of the type it took as an input
tune_model <- function(model_type) {
  tune_method <- trainControl(method = "repeatedcv", number = 10, repeats = 5, selectionFunction = "best")
  model <- train(mpg_binary ~ ., train_set, method = model_type, trControl = tune_method)
  return(model)
}

# tune the KNN model
tuned_knn_mod <- tune_model("knn")
tuned_knn_mod

# predict for test set with new tuned model
test_set$knn_tuned_prediction <- predict(tuned_knn_mod, test_set)
test_set$knn_tuned_accuracy <- ifelse(test_set$knn_tuned_prediction == test_set$mpg_binary, "correct", "incorrect")

# accuracy rate:
table_knn_tuned <- prop.table(table(test_set$knn_tuned_accuracy)) %>% print() %>% unname()

# Tune the Decision Tree model
tuned_dt_mod <- tune_model("C5.0")
tuned_dt_mod

# predict for test set
test_set$dt_tuned_prediction <- predict(tuned_dt_mod, test_set)
test_set$dt_tuned_accuracy <- ifelse(test_set$dt_tuned_prediction == test_set$mpg_binary, "correct", "incorrect")

# accuracy rate
table_dt_tuned <- prop.table(table(test_set$dt_tuned_accuracy)) %>% print() %>% unname()

# compile accuracy rates for all four models
results_table_2 <- rbind(table_knn_tuned, table_dt_tuned) %>% data.frame() 
names(results_table_2) <- c("correct", "incorrect")
results_table_2$model <- c("knn_tuned", "dt_tuned")
results_table <- rbind(results_table, results_table_2)

# plot accuracy for all models
test_long <- to_long_format(c("knn_accuracy", "knn_tuned_accuracy", "dt_accuracy", "dt_tuned_accuracy"))
accuracy_plot()

#### Secition 3 ####

# partition dataset into 10 folds
set.seed(123)
folds <- createFolds(dat$mpg_binary, k = 10)

# this function performs a 10-fold cross-validation using either
# a k-nearest neighbors or decision tree model based on a 'model_type'
# global variable.
#   it takes:
#     a subset (vector of indices) of the overall data
#   it trains the model on all but the subset 'x', predicts for x
#   and returns:
#     an accuracy rate for that fold
cross_validation <- function(x) {
  training <- dat[-x, ]
  testing <- dat[x, ]
  if (model_type == 'knn') {
    testing$prediction <- knn(training, testing, training$mpg_binary, k = tuned_knn_mod$bestTune$k)
    accuracy <- ifelse(testing$prediction == testing$mpg_binary, 1, 0)
  }
  else if (model_type == 'C5.0') {
    dt_model <- C5.0(mpg_binary ~ ., data = training, trials = tuned_dt_mod$bestTune[1,1], winnow = FALSE, rules = FALSE)
    testing$prediction <- predict(dt_model, testing)
    accuracy <- ifelse(testing$prediction == testing$mpg_binary, 1, 0)
  }
  return(sum(accuracy) / length(accuracy))
}

# knn cross-validation
model_type  <- 'knn'
cv_knn <- ldply(folds, cross_validation)
cv_knn$model <- model_type

# decision-tree cross-validation
model_type <- 'C5.0'
cv_dt <- ldply(folds, cross_validation)
cv_dt$model <- 'dt'

# bind results together into a data frame, adjust variables and summarize
# average/sd error per model
cv_results <- rbind(cv_knn, cv_dt)
cv_results$fold <- cv_results$.id %>% as.factor() %>% as.numeric()
cv_results <- ddply(cv_results, ~model, summarise, avg_acc = round(mean(V1), 3), 
                       sd_acc = round(sd(V1), 3)) %>% join(cv_results, ., by = 'model')

# plot the dataset using a time series with point ranges indicating the standard 
# deviation for each model
ggplot(cv_results, aes(x = fold, y = V1, ymin = (V1 - sd_acc), ymax = (V1 + sd_acc), color = model)) + 
  geom_hline(aes(yintercept = avg_acc), linetype = 'dashed', alpha = 0.5) +
  geom_line() + geom_pointrange() +
  facet_wrap(~model, scales = 'fixed', ncol = 1) + 
  ggtitle("10-Fold Cross-Validation Results") + 
  ylab("predictive accuracy rate") + xlab("fold #") +
  geom_text(data = cv_results[c(1,20), ], 
            aes(x = 7.5, y = .78, label = paste("avg. accuracy: ", avg_acc, sep = ""), hjust = "left")) +
  geom_text(data = cv_results[c(1,20), ], 
            aes(x = 7.5, y = .75, label = paste("st. dev. accuracy: ", sd_acc, sep = ""), hjust = "left")) +
  geom_text(data = cv_results[c(1,20), ], 
            aes(x = .5, y = avg_acc + .01, label = paste('avg. accuracy'), hjust = "left", vjust = "below"), color = 'black') +
  scale_x_continuous(breaks = c(1:10), limits = c(0.5, 10)) +
  plot_theme()
