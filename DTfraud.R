library(party)
library(gmodels)
library(tree)
library(caret)

FraudCheck <- read.csv(file.choose())
hist(FraudCheck$Taxable.Income)

Risky_Good = ifelse(FraudCheck$Taxable.Income<= 30000, "Risky", "Good")
FC = data.frame(FraudCheck,Risky_Good)
table(FC$Risky_Good)

FC_train <- FC[1:400,]
FC_test <- FC[401:600,]

op_tree = ctree(Risky_Good~.-Taxable.Income, data=FC_train)
summary(op_tree)
plot(op_tree)
pred_test_df <- predict(op_tree,newdata=FC_test)
mean(pred_test_df==FC_test$Risky_Good)
confusionMatrix(pred_test_df,FC_test$Risky_Good)

fc_tree = tree(Risky_Good~.-Taxable.Income, data=FC_train)
summary(fc_tree)
plot(fc_tree)
text(fc_tree,pretty = 0)

pred_tree <- as.data.frame(predict(fc_tree,newdata=FC_test))
pred_test_df <- predict(fc_tree,newdata=FC_test)
pred_tree$final <- colnames(pred_test_df)[apply(pred_test_df,1,which.max)]
pred_tree$final <- as.factor(pred_tree$final)
mean(pred_tree$final==FC_test$Risky_Good)
confusionMatrix(pred_tree$final,FC_test$Risky_Good)
