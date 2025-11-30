##################################################
# FBA Group Project
##################################################

# Load required libraries
library(tidyverse)
library(lubridate)
library(janitor)
library(scales)
library(data.table)
library(ggplot2)
library(tidyr)
library(readr)
library(skimr)
library(ggcorrplot)
library(dplyr)

# Load data from CSV file (adjust path as needed)
bank_data <- read_csv2("bank-full.csv")
# Data Preview
head(bank_data)
dim(bank_data)
skim(bank_data)

# Convert appropriate columns to factors (categorical variables)
cat_vars <- c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y")
bank_data[cat_vars] <- lapply(bank_data[cat_vars], as.factor)

# Make sure the target variable is binary factor with 'no' as reference and 'yes' as level of interest
bank_data$y <- relevel(bank_data$y, ref = "no")


# ----------------------------------------
# Data Partitioning: 70% Training, 30% Validation
# ----------------------------------------
set.seed(123)  # for reproducibility
n <- nrow(bank_data)
train_idx <- sample(seq_len(n), size = 0.7 * n)

train_data <- bank_data[train_idx, ]
valid_data <- bank_data[-train_idx, ]

# ----------------------------------------
# Fit Models on Training Set
# ----------------------------------------

# Marketing Factors Model
model_marketing <- glm(y ~ day + month + duration + campaign + pdays + previous + poutcome,
                       data = train_data, family = binomial)
summary(model_marketing)

# Financial Factors Model
model_financial <- glm(y ~ balance + loan + housing,
                       data = train_data, family = binomial)
summary(model_financial)

# Demographical Factors Model
model_demographic <- glm(y ~ age + education + marital + job,
                         data = train_data, family = binomial)
summary(model_demographic)
# ----------------------------------------
# Odds Ratios and Confidence Intervals
# ----------------------------------------
exp(cbind(OR = coef(model_marketing), confint(model_marketing)))
exp(cbind(OR = coef(model_financial), confint(model_financial)))
exp(cbind(OR = coef(model_demographic), confint(model_demographic)))

# ----------------------------------------
# Model Accuracy & Pseudo R2 (training set)
# ----------------------------------------
if(!require(pscl)) install.packages("pscl"); library(pscl)
pR2(model_marketing)
pR2(model_financial)
pR2(model_demographic)

# ----------------------------------------
# Training Set: Confusion Matrices
# ----------------------------------------
pred_marketing_train <- predict(model_marketing, type = 'response')
table(train_data$y, pred_marketing_train > 0.5)
pred_financial_train <- predict(model_financial, type = 'response')
table(train_data$y, pred_financial_train > 0.5)
pred_demographic_train <- predict(model_demographic, type = 'response')
table(train_data$y, pred_demographic_train > 0.5)

# ----------------------------------------
# ROC Curve and AUC (training set)
# ----------------------------------------
library(ROCR)
library(pROC)      # For AUC calculation

# Marketing model
roc_marketing_train <- roc(train_data$y, pred_marketing_train)
cat("Marketing Model AUC (train):", auc(roc_marketing_train), "\n")
plot(roc_marketing_train, main = paste("Marketing Model ROC (AUC =", round(auc(roc_marketing_train), 3), ")"))

# Financial model
roc_financial_train <- roc(train_data$y, pred_financial_train)
cat("Financial Model AUC (train):", auc(roc_financial_train), "\n")
plot(roc_financial_train, main = paste("Financial Model ROC (AUC =", round(auc(roc_financial_train), 3), ")"))

# Demographic model
roc_demographic_train <- roc(train_data$y, pred_demographic_train)
cat("Demographic Model AUC (train):", auc(roc_demographic_train), "\n")
plot(roc_demographic_train, main = paste("Demographic Model ROC (AUC =", round(auc(roc_demographic_train), 3), ")"))

# ----------------------------------------
# Model Validation: On Validation Set
# ----------------------------------------

# Marketing
pred_marketing_valid <- predict(model_marketing, newdata=valid_data, type="response")
table(valid_data$y, pred_marketing_valid > 0.5)
roc_marketing_valid <- roc(valid_data$y, pred_marketing_valid)
cat("Marketing Model AUC (validation):", auc(roc_marketing_valid), "\n")
plot(roc_marketing_valid, main = paste("Marketing Model ROC (Validation AUC =", round(auc(roc_marketing_valid), 3), ")"))

# Financial
pred_financial_valid <- predict(model_financial, newdata=valid_data, type="response")
table(valid_data$y, pred_financial_valid > 0.5)
roc_financial_valid <- roc(valid_data$y, pred_financial_valid)
cat("Financial Model AUC (validation):", auc(roc_financial_valid), "\n")
plot(roc_financial_valid, main = paste("Financial Model ROC (Validation AUC =", round(auc(roc_financial_valid), 3), ")"))

# Demographic
pred_demographic_valid <- predict(model_demographic, newdata=valid_data, type="response")
table(valid_data$y, pred_demographic_valid > 0.5)
roc_demographic_valid <- roc(valid_data$y, pred_demographic_valid)
cat("Demographic Model AUC (validation):", auc(roc_demographic_valid), "\n")
plot(roc_demographic_valid, main = paste("Demographic Model ROC (Validation AUC =", round(auc(roc_demographic_valid), 3), ")"))

#------------------------------------------------------------
# Create predicted probabilities on validation set for each factor group
valid_data$pred_marketing <- predict(model_marketing, newdata = valid_data, type = "response")
valid_data$pred_financial <- predict(model_financial, newdata = valid_data, type = "response")
valid_data$pred_demographic <- predict(model_demographic, newdata = valid_data, type = "response")

# Gather for comparison
library(tidyr)
factor_preds <- valid_data %>%
  select(y, pred_marketing, pred_financial, pred_demographic) %>%
  pivot_longer(
    cols = starts_with("pred_"), 
    names_to = "Factor_Group", 
    values_to = "Predicted_Prob"
  ) %>%
  mutate(
    Factor_Group = case_when(
      Factor_Group == "pred_marketing" ~ "Marketing",
      Factor_Group == "pred_financial" ~ "Financial",
      Factor_Group == "pred_demographic" ~ "Demographic"
    )
  )

# Plot average predicted probability for each factor group
library(ggplot2)
ggplot(factor_preds, aes(x = Factor_Group, y = Predicted_Prob, fill = Factor_Group)) +
  stat_summary(fun = mean, geom = "bar", color = "black") +
  labs(title = "Average Predicted Subscription Probability by Factor Group",
       x = "Factor Group", y = "Average Predicted Probability of Subscription") +
  theme_minimal()

library(broom)
# Marketing model
marketing_coef <- tidy(model_marketing, conf.int = TRUE, exponentiate = TRUE)
marketing_coef$Factor_Group <- "Marketing"
# Financial model
financial_coef <- tidy(model_financial, conf.int = TRUE, exponentiate = TRUE)
financial_coef$Factor_Group <- "Financial"
# Demographic model
demographic_coef <- tidy(model_demographic, conf.int = TRUE, exponentiate = TRUE)
demographic_coef$Factor_Group <- "Demographic"

# Combine all for one plot
all_coef <- bind_rows(marketing_coef, financial_coef, demographic_coef)

library(purrr)

# Filter out intercept and plot
plots <- all_coef %>%
  filter(term != "(Intercept)") %>%
  group_split(Factor_Group) %>%
  map(~ ggplot(.x, aes(x = reorder(term, estimate), y = estimate)) +
        geom_point(size = 4, color = "steelblue") +
        geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.25) +
        geom_hline(yintercept = 1, linetype = "dashed") +
        coord_flip() +
        labs(title = paste(unique(.x$Factor_Group), "Factors: Odds Ratio for Subscription"),
             x = "Predictor", y = "Odds Ratio (exp(Coefficient))")
  )
# Access plots individually
plots[[1]]  # Marketing
plots[[2]]  # Financial
plots[[3]]  # Demographic


all_coef %>%
  filter(term != "(Intercept)") %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate, color = Factor_Group)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.25) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  coord_flip() +
  labs(title = "Individual Factors: Odds Ratio for Subscription",
       x = "Predictor", y = "Odds Ratio (exp(Coefficient))") +
  facet_wrap(~Factor_Group, scales = "free_y")

library(broom)

# Marketing Factors
marketing_table <- tidy(model_marketing, conf.int = TRUE, exponentiate = TRUE)
# Financial Factors
financial_table <- tidy(model_financial, conf.int = TRUE, exponentiate = TRUE)
# Demographic Factors
demographic_table <- tidy(model_demographic, conf.int = TRUE, exponentiate = TRUE)

# View as data frames
print(marketing_table)
print(financial_table)
print(demographic_table)

