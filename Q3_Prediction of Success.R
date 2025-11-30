# Load Libraries

library(caret)
library(pROC)
library(ModelMetrics)
library(dplyr)

set.seed(123)  

# Load & Prepare Data

library(readxl)
df <- read_csv("C:\\Users\\Yuvan Rahul\\OneDrive\\Documents\\bank\\bank-full.csv")
library(skimr)

skim(df)

df$y <- as.factor(df$y)  


# Split into Train/Test

train_index <- createDataPartition(df$y, p = 0.8, list = FALSE)
train <- df[train_index, ]
test <- df[-train_index, ]


# Define Training Control

ctrl <- trainControl(
  method = "cv",               
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  
  savePredictions = "final"
)

levels(train$y)

# caret expects the positive class to be "yes" or "1" listed second

train$y <- relevel(train$y, ref = "no")


# Train Multiple Models

# Logistic Regression
model_lr <- train(y ~ ., data = train,
                  method = "glm",
                  metric = "ROC",
                  trControl = ctrl,
                  family = "binomial")

# Random Forest
model_rf <- train(y ~ ., data = train,
                  method = "rf",
                  metric = "ROC",
                  trControl = ctrl)

# Decision Tree (CART)
model_dt <- train(y ~ ., data = train,
                  method = "rpart",
                  metric = "ROC",
                  trControl = ctrl)

# K-Nearest Neighbors
model_knn <- train(y ~ ., data = train,
                   method = "knn",
                   metric = "ROC",
                   trControl = ctrl)




# Compare Models by Resampling

results <- resamples(list(
  Logistic = model_lr,
  RandomForest = model_rf,
  DecisionTree = model_dt,
  KNN = model_knn
  
))

# Summary of performance metrics
summary(results)

# Visual comparison
bwplot(results, metric = "ROC", main = "ROC-AUC Comparison")
dotplot(results, metric = "ROC", main = "ROC-AUC Comparison")


#testing levels
levels(as.factor(test$y))

test$y <- factor(test$y, levels = c("no", "yes"))

# Ensure probabilities are numeric
# test$prob <- as.numeric(test$prob)


# Evaluate Models on Test Set

models <- list(
  Logistic = model_lr,
  RandomForest = model_rf,
  DecisionTree = model_dt,
  KNN = model_knn
  )

test_results <- data.frame(Model = character(), Accuracy = numeric(),
                           ROC_AUC = numeric(), F1 = numeric(), stringsAsFactors = FALSE)

for (name in names(models)) {
  mod <- models[[name]]
  prob <- predict(mod, newdata = test, type = "prob")[,"yes"]
  pred <- predict(mod, newdata = test)
  
  roc_obj <- pROC::roc(response = test$y,
                       predictor = prob,
                       levels = c("no", "yes"),
                       direction = "<")
  auc_val <- pROC::auc(roc_obj)
  
  cm <- caret::confusionMatrix(pred, test$y, positive = "yes")
  
  acc <- cm$overall["Accuracy"]
  f1 <- cm$byClass["F1"]
  
  test_results <- rbind(test_results,
                        data.frame(Model = name, Accuracy = acc,
                                   ROC_AUC = auc_val, F1 = f1))
}


# Display Comparison

test_results <- test_results %>% arrange(desc(ROC_AUC))
print(test_results)


test_predictions <- data.frame(
  Actual = test$y,
  Predicted = pred,
  Probability_Yes = prob
)


# adding probabilities to the df
head(test_predictions)


df$rf_prob_yes <- predict(model_rf, newdata = df, type = "prob")[,"yes"]
df$rf_predicted_class <- predict(model_rf, newdata = df)



#roc curve
library(ggplot2)

plot_roc <- function(model, name) {
  prob <- predict(model, newdata = test, type = "prob")[,"yes"]
  roc_obj <- roc(test$y, prob)
  data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    model = name
  )
}

roc_df <- rbind(
  plot_roc(model_lr, "Logistic"),
  plot_roc(model_rf, "RandomForest"),
  plot_roc(model_dt, "Decision Tree"),
  plot_roc(model_knn, "KNN")
)

ggplot(roc_df, aes(x = fpr, y = tpr, color = model)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed") +
  labs(title = "ROC Curves for All Models",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal()


# Save the dataframe with predictions to a CSV file
write.csv(df, "C:\\Users\\Yuvan Rahul\\OneDrive\\Documents\\bank\\df_with_predictions.csv",
          row.names = FALSE)

