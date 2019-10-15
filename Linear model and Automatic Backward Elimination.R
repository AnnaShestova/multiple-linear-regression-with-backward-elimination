# Importing dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'), 
                       labels = c(1, 2, 3))

# Splitting dataset into training set an test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
#                   data = training_set)

# Applying linear model
regressor = lm(formula = Profit ~ .,
               data = trainig_set)
summary(regressor)

# Predicting test set results
y_pred = predict(regressor, newdata = test_set)

# Creating automated backward elimination function
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

# Applying backward eliminated function
SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)

# Applying linear model to selected features
regressor_new = lm(formula = Profit ~ R.D.Spend,
               data = trainig_set)
summary(regressor_new)

# Predicting test set results
y_pred = predict(regressor_new, newdata = test_set)

# Visualizing test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$R.D.Spend, y = test_set$Profit),
             colour = 'red') +
  geom_line(aes(x = training_set$R.D.Spend, y = predict(regressor_new, newdata = training_set)),
            colour = 'blue') +
  ggtitle('R.D.Spend vs Profit (Test set)') +
  xlab('R.D.Spend') +
  ylab('Profit')
