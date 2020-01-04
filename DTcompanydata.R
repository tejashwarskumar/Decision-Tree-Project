library(party)
library(caret)
library(tree)

CompanyData <- read.csv(file.choose())
hist(CompanyData$Sales)

High = ifelse(CompanyData$Sales<10, "No", "Yes")
CD = data.frame(CompanyData, High)
table(CD$High)

CD_train <- CD[1:200,]
CD_test <- CD[201:400,]

op_tree = ctree(High ~ .-Sales, data = CD_train)
summary(op_tree)
plot(op_tree)
pred_test_df <- predict(op_tree,newdata=CD_test)
mean(pred_test_df==CD_test$High)
confusionMatrix(CD_test$High,pred_test_df)

cd_tree <- tree(High~.-Sales, data=CD_train)
summary(cd_tree)
plot(cd_tree)
text(cd_tree,pretty = 0)

pred_tree <- as.data.frame(predict(cd_tree,newdata=CD_test))
pred_test_df <- predict(cd_tree,newdata=CD_test)
pred_tree$final <- colnames(pred_test_df)[apply(pred_test_df,1,which.max)]
pred_tree$final <- as.factor(pred_tree$final)
mean(pred_tree$final==CD_test$High)
confusionMatrix(CD_test$High,pred_tree$final)
