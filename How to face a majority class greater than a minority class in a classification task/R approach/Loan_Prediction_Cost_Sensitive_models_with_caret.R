# Prepare Workspace
suppressWarnings({library(ggplot2)})
suppressWarnings({library(tidyverse)})
suppressWarnings({library(caret)})
suppressWarnings({library(corrplot)})
suppressWarnings({library(gridExtra)})
suppressWarnings({library(MLmetrics)})

# Upload Dataset
path <- "C:/Users/user/Documents/eRUM2020/hmeq.csv"
df = read.csv(path)

# Dimensions of data set
dim(df)
# List types for each attribute
sapply(df, class)
# Take a peek at the first rows of the data set
head(df,5)
# Summarize attribute distributions
summary(df)
# Summarize data structure
str(df)

# Formatting Features and managing some levels
BAD <- df$BAD <- as.factor(df$BAD)
df$LOAN <- as.numeric(df$LOAN)
df$DEROG <- as.factor(df$DEROG)
df$DELINQ <- as.factor(df$DELINQ)
df$NINQ <- as.factor(df$NINQ)
df$CLNO <- as.factor(df$CLNO)
df$JOB[df$JOB == ""] <- "NA"
df$REASON[df$REASON == ""] <- "NA"

# Check missing values
mi_summary <- function(data_frame){
  mi_summary<-c()
  for (col in colnames(data_frame)){
    mi_summary <- c(mi_summary,mean(is.na(data_frame[,col])*100))
  }
  mi_summary_new <- mi_summary[mi_summary>0]
  mi_summary_cols <- colnames(data_frame)[mi_summary>0]
  mi_summary <- data.frame('col_name' = mi_summary_cols, 'perc_missing' = mi_summary_new)
  mi_summary <- mi_summary[order(mi_summary[,2], decreasing = TRUE), ]
  mi_summary[,2] <- round(mi_summary[,2],6)
  rownames(mi_summary) <- NULL
  return(mi_summary)
}
missing_summary <- mi_summary(df)
missing_summary

# Input boolean variables for features with NA's 
df <- df %>%
  mutate(DEBTINC_NA = ifelse(is.na(DEBTINC),1,0)) %>%
  mutate(DEROG_NA = ifelse(is.na(DEROG),1,0)) %>%
  mutate(DELINQ_NA = ifelse(is.na(DELINQ),1,0)) %>%
  mutate(MORTDUE_NA = ifelse(is.na(MORTDUE),1,0)) %>%
  mutate(YOJ_NA = ifelse(is.na(YOJ),1,0)) %>%
  mutate(NINQ_NA = ifelse(is.na(NINQ),1,0)) %>%
  mutate(CLAGE_NA = ifelse(is.na(CLAGE),1,0)) %>%
  mutate(CLNO_NA = ifelse(is.na(CLNO),1,0)) %>%
  mutate(VALUE_NA = ifelse(is.na(VALUE),1,0)) %>%
  mutate(JOB_NA = ifelse(is.na(JOB),1,0)) %>%
  mutate(REASON_NA = ifelse(is.na(REASON),1,0))

# Input missing values with median for numerical columns and with the most common level for categorical columns
for (col in missing_summary$col_name){
  if (class(df[,col]) == 'factor'){
    unique_levels <- unique(df[,col])
    df[is.na(df[,col]), col] <- unique_levels[which.max(tabulate(match(df[,col], unique_levels)))]
  } else {
    df[is.na(df[,col]),col] <- median(as.numeric(df[,col]), na.rm = TRUE)
  }
}

# Check results
pMiss <- function(x){sum(is.na(x))/length(x)*100}
pMiss <- apply(df,2,pMiss)
pMiss <- pMiss[pMiss > 0]
pMiss <- pMiss[order(pMiss, decreasing=T)]
pMiss

# Formatting new features and managing some levels
df$DEROG_NA <- as.factor(df$DEROG_NA)
df$DEBTINC_NA <- as.factor(df$DEBTINC_NA)
df$DELINQ_NA <- as.factor(df$DELINQ_NA)
df$MORTDUE_NA <- as.factor(df$MORTDUE_NA)
df$YOJ_NA <- as.factor(df$YOJ_NA)
df$NINQ_NA <- as.factor(df$NINQ_NA)
df$CLAGE_NA <- as.factor(df$CLAGE_NA)
df$CLNO_NA <- as.factor(df$CLNO_NA)
df$VALUE_NA <- as.factor(df$VALUE_NA)
df$JOB_NA <- as.factor(df$JOB_NA)
df$REASON_NA <- as.factor(df$REASON_NA)
df$JOB <- factor(df$JOB, labels=c('Mgr','Office','Other','ProfExe','Sales','Self'))
df$REASON <- factor(df$REASON, labels=c('DebtCon','HomeImp'))


# Split data set into categorical, boolean and numerical variables
cat <- df[,sapply(df, is.factor)] %>%
  select_if(~nlevels(.) <=15 ) %>%
  select(-BAD)
bol <- df[,c('DEBTINC_NA','DEROG_NA','DELINQ_NA','MORTDUE_NA','YOJ_NA','NINQ_NA','CLAGE_NA','CLNO_NA','VALUE_NA','JOB_NA','REASON_NA')]
num <- df[,sapply(df, is.numeric)]

# Summarize the class distribution of the target variable
cbind(freq=table(df$BAD), percentage=prop.table(table(df$BAD))*100)
# Visualize data
ggplot(df, aes(BAD, fill=BAD)) + geom_bar() +
  scale_fill_brewer(palette = "Set1") +
  ggtitle("Distribution of Target variable")

# Analysis for categorical features (barplot, univariate analysis, bivariate analysis)
# Univariate Analysis
cat <- cat[,c('DELINQ','REASON','JOB','DEROG')]
for(i in 1:length(cat)) {
  counts <- table(cat[,i])
  name <- names(cat)[i]
  barplot(counts, main=name, col=c("blue","red","green","orange","purple"))
}

# Bivariate Analysis with Feature Selection Analysis
par(mfrow=c(2,2))
for(i in 1:length(cat)){
  freq=table(cat[,i])
  percentage=prop.table(table(cat[,i]))*100
  freq_cat_outcome=table(BAD,cat[,i])
  name <- names(cat)[i]
  cat(sep="\n")
  cat(paste("Distribution of", name), sep="\n")
  print(cbind(freq,percentage))
  cat(sep="\n")
  cat(paste("Distribution by Target variable and", name), sep="\n")
  print(freq_cat_outcome)
  cat(sep="\n")
  cat(paste("Chi-squared test by Target variable and", name), sep="\n")
  suppressWarnings({print(chisq.test(table(BAD,cat[,i])))})
}

# Visualization of Bivariate Analysis
pl1 <- cat %>%
  ggplot(aes(x=BAD, y=DELINQ, fill=BAD)) + 
  geom_bar(stat='identity') + 
  ggtitle("Distribution by BAD and DELINQ")
pl2 <- cat %>%
  ggplot(aes(x=BAD, y=REASON, fill=BAD)) + 
  geom_bar(stat='identity') +
  ggtitle("Distribution by BAD and REASON")
pl3 <- cat %>%
  ggplot(aes(x=BAD, y=JOB, fill=BAD)) + 
  geom_bar(stat='identity') +
  ggtitle("Distribution by BAD and JOB")
pl4 <- cat %>%
  ggplot(aes(x=BAD, y=DEROG, fill=BAD)) + 
  geom_bar(stat='identity') +
  ggtitle("Distribution by BAD and DEROG")
par(mfrow=c(2,2))
grid.arrange(pl1,pl2,pl3,pl4, ncol=2)

# One-hot encoding on categorical features
dmy <- dummyVars("~.", data = cat,fullRank = F)
cat_num <- data.frame(predict(dmy, newdata = cat))
# Remove correlated levels from boolean features
drop_cols <- c('DEBTINC_NA.0','DEROG_NA.0','DELINQ_NA.0','MORTDUE_NA.0','YOJ_NA.0','NINQ_NA.0','CLAGE_NA.0','CLNO_NA.0','VALUE_NA.0','JOB_NA.0','REASON_NA.0')
categorical <- cat_num[,!colnames(cat_num) %in% drop_cols]

# Analysis for numerical features (univariate analysis, bivariate analysis)
# Univariate Analysis, histograms
par(mfrow=c(2,3))
for(i in 1:length(num)) {
  hist(num[,i], main=names(num)[i], col='blue')
}

# Univariate Analysis, boxplots
par(mfrow=c(2,3))
for(i in 1:length(num)) {
  boxplot(num[,i], main=names(num)[i], col='orange')
}

# Univariate Analysis, densityplots
par(mfrow=c(2,3))
for(i in 1:length(num)){
  plot(density(num[,i]), main=names(num)[i], col='red')
}

# Bivariate Analysis
for(i in 1:length(num)){
  name <- names(num)[i]
  cat(paste("Distribution of", name), sep="\n")
  #cat(names(num)[i],sep = "\n")
  print(summary(num[,i]))
  cat(sep="\n")
  stand.deviation = sd(num[,i])
  variance = var(num[,i])
  skewness = mean((num[,i] - mean(num[,i]))^3/sd(num[,i])^3)
  kurtosis = mean((num[,i] - mean(num[,i]))^4/sd(num[,i])^4) - 3
  outlier_values <- sum(table(boxplot.stats(num[,i])$out))
  cat(paste("Statistical analysis of", name), sep="\n")
  print(cbind(stand.deviation, variance, skewness, kurtosis, outlier_values))
  cat(sep="\n")
  cat(paste("anova_test between BAD and", name),sep = "\n")
  print(summary(aov(as.numeric(BAD)~num[,i], data=num)))
  cat(sep="\n")
}

# Visualization of Bivariate Analysis
pl5 <- num %>%
  ggplot(aes(x=BAD, y=LOAN, fill=BAD)) + geom_boxplot() 
pl6 <- num %>%
  ggplot(aes(x=BAD, y=MORTDUE, fill=BAD)) + geom_boxplot() 
pl7 <- num %>%
  ggplot(aes(x=BAD, y=VALUE, fill=BAD)) + geom_boxplot() 
pl8 <- num %>%
  ggplot(aes(x=BAD, y=YOJ, fill=BAD)) + geom_boxplot() 
pl9 <- num %>%
  ggplot(aes(x=BAD, y=CLAGE, fill=BAD)) + geom_boxplot() 
pl10 <- num %>%
  ggplot(aes(x=BAD, y=DEBTINC, fill=BAD)) + geom_boxplot() 
par(mfrow=c(2,3))
grid.arrange(pl5,pl6,pl7,pl8,pl9,pl10, ncol=2)

# Handling outliers
# Before
ggplot(num, aes(x = LOAN, fill = BAD)) + geom_density(alpha = .3) + ggtitle("LOAN")
# Managing outliers
qnt <- quantile(num$LOAN, probs=c(.25, .75), na.rm = T)
caps <- quantile(num$LOAN, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(num$LOAN, na.rm = T)
num$LOAN[num$LOAN < (qnt[1] - H)]  <- caps[1]
num$LOAN[num$LOAN >(qnt[2] + H)] <- caps[2]
# After
ggplot(num, aes(x = LOAN, fill = BAD)) + geom_density(alpha = .3) + ggtitle("LOAN after handled outliers")
# Before
ggplot(num, aes(x = MORTDUE, fill = BAD)) + geom_density(alpha = .3) + ggtitle("MORTDUE")
# Managing outliers
qnt <- quantile(num$MORTDUE, probs=c(.25, .75), na.rm = T)
caps <- quantile(num$MORTDUE, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(num$MORTDUE, na.rm = T)
num$MORTDUE[num$MORTDUE < (qnt[1] - H)]  <- caps[1]
num$MORTDUE[num$MORTDUE >(qnt[2] + H)] <- caps[2]
# After
ggplot(num, aes(x = MORTDUE, fill = BAD)) + geom_density(alpha = .3)  + ggtitle("MORTDUE after handled outliers")
# Before
ggplot(num, aes(x = VALUE, fill = BAD)) + geom_density(alpha = .3) + ggtitle("VALUE")
# Managing outliers
qnt <- quantile(num$VALUE, probs=c(.25, .75), na.rm = T)
caps <- quantile(num$VALUE, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(num$VALUE, na.rm = T)
num$VALUE[num$VALUE < (qnt[1] - H)]  <- caps[1]
num$VALUE[num$VALUE >(qnt[2] + H)] <- caps[2]
# After
ggplot(num, aes(x = VALUE, fill = BAD)) + geom_density(alpha = .3) + ggtitle("VALUE after handled outliers")
# Before
ggplot(num, aes(x = YOJ, fill = BAD)) + geom_density(alpha = .3) + ggtitle("YOJ")
# Managing outliers
qnt <- quantile(num$YOJ, probs=c(.25, .75), na.rm = T)
caps <- quantile(num$YOJ, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(num$YOJ, na.rm = T)
num$YOJ[num$YOJ < (qnt[1] - H)]  <- caps[1]
num$YOJ[num$YOJ >(qnt[2] + H)] <- caps[2]
# After
ggplot(num, aes(x = YOJ, fill = BAD)) + geom_density(alpha = .3) + ggtitle("YOJ after handled outliers")
# Before
ggplot(num, aes(x = CLAGE, fill = BAD)) + geom_density(alpha = .3) + ggtitle("CLAGE")
# Managing outliers
qnt <- quantile(num$CLAGE, probs=c(.25, .75), na.rm = T)
caps <- quantile(num$CLAGE, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(num$CLAGE, na.rm = T)
num$CLAGE[num$CLAGE < (qnt[1] - H)]  <- caps[1]
num$CLAGE[num$CLAGE >(qnt[2] + H)] <- caps[2]
# After
ggplot(num, aes(x = CLAGE, fill = BAD)) + geom_density(alpha = .3) + ggtitle("CLAGE after handled outliers")
# Before
ggplot(num, aes(x = DEBTINC, fill = BAD)) + geom_density(alpha = .3) + ggtitle("DEBTINC")
# Managing outliers
qnt <- quantile(num$DEBTINC, probs=c(.25, .75), na.rm = T)
caps <- quantile(num$DEBTINC, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(num$DEBTINC, na.rm = T)
num$DEBTINC[num$DEBTINC < (qnt[1] - H)]  <- caps[1]
num$DEBTINC[num$DEBTINC >(qnt[2] + H)] <- caps[2]
# After
ggplot(num, aes(x = DEBTINC, fill = BAD)) + geom_density(alpha = .3) + ggtitle("DEBTINC after handled outliers")

# Delete Zero-and Near Zero-Variance Predictors
data <- cbind(categorical,num) 
nzv <- nearZeroVar(data, saveMetrics= TRUE)
nzv[nzv$nzv,][1:15,]
nzv <- nearZeroVar(data)
data_new <- data[, -nzv]

# Correlation
# Visualization
par(mfrow=c(1,1))
cor <- cor(data_new,use="complete.obs",method = "spearman")
corrplot(cor, type="lower", tl.col = "black", diag=FALSE, method="number", mar = c(0, 0, 2, 0), title="Correlation") 
summary(cor[upper.tri(cor)])

# Delete correlated features
tmp <- cor(data_new)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0
df_new <- data_new[,!apply(tmp,2,function(x) any(abs(x) > 0.75))]
cor <- cor(df_new,use="complete.obs",method = "spearman")
summary(cor[upper.tri(cor)])

# Pre-processing

# calculate the pre-process parameters from the data set
set.seed(2019)
preprocessParams <- preProcess(df_new, method=c("center", "scale"))
# Transform the data set using the parameters
transformed <- predict(preprocessParams, df_new)
# Manage levels on the target variable
y <- as.factor(df$BAD)
transformed <- cbind.data.frame(transformed,y)
levels(transformed$y) <- make.names(levels(factor(transformed$y)))
str(transformed)

# Split data set
# Draw a random, stratified sample including p percent of the data
set.seed(12345)
test_index <- createDataPartition(transformed$y, p=0.80, list=FALSE)
# select 20% of the data for test
itest <- transformed[-test_index,]
# use the remaining 80% of data to training the models
itrain <- transformed[test_index,]

# Cost-sensitive learning: evaluating models by Caret

# Stratified cross-validation
folds <- 5
set.seed(2019)
cvIndex <- createFolds(factor(itrain$y), folds, returnTrain = T)
control <- trainControl(index = cvIndex,method="repeatedcv", number=folds,classProbs = TRUE, 
                        summaryFunction=prSummary)
metric <- "AUC"


# REGULARIZED RANDOM FOREST
set.seed(2019)
fit.rrf <- train(y~., data=itrain, method="RRF", metric=metric, trControl=control)
print(fit.rrf)
plot(varImp(fit.rrf),15, main='Regularized Random Forest feature selection')

# Penalized Multinomial Regression
set.seed(2019)
fit.pmr <- train(y~., data=itrain, method="multinom", metric=metric, trControl=control)
print(fit.pmr)
plot(varImp(fit.pmr),15, main = 'Penalized Multinomial Regression feature selection')


# Comparison of algorithms
results <- resamples(list(rrf=fit.rrf, pmr=fit.pmr))
cat(paste('Results'), sep='\n')
summary(results)
par(mar = c(4, 11, 1, 1))
dotplot(results, main = 'AUC results from algorithms')


# Predictions
par(mfrow=c(2,2))
set.seed(2019)
prediction.rrf <-predict(fit.rrf,newdata=itest,type="raw")
set.seed(2019)
prediction.pmr <-predict(fit.pmr,newdata=itest,type="raw")


# Visualization of results
cat(paste('Confusion Matrix Regularized Random Forest Model'), sep='\n')
confusionMatrix(prediction.rrf, itest$y)
F1_train <- F1_Score(itrain$y, predict(fit.rrf,newdata=itrain,type="raw"))
F1_test <- F1_Score(itest$y, prediction.rrf)
cat(paste('F1_train_rrf:',F1_train,'F1_test_rrf:', F1_test), sep='\n')
cat(paste('Confusion Matrix Penalized Multinomial Regression Model'), sep='\n')
confusionMatrix(prediction.pmr, itest$y)
F1_train <- F1_Score(itrain$y, predict(fit.pmr,newdata=itrain,type="raw"))
F1_test <- F1_Score(itest$y, prediction.pmr)
cat(paste('F1_train_pmr:',F1_train,'F1_test_pmr:', F1_test), sep='\n')
cat(paste('Confusion Matrix LSVM Model'), sep='\n')


# Confusion Matrix Plots
par(mfrow=c(2,2))
ctable.rrf <- table(prediction.rrf, itest$y)
fourfoldplot(ctable.rrf, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "RRF Confusion Matrix")
ctable.pmr <- table(prediction.pmr, itest$y)
fourfoldplot(ctable.pmr, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "PMR Confusion Matrix")

