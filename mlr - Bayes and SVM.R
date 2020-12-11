library(mlr)
library(tidyverse)

install.packages("mlbench")
data(HouseVotes84, package = "mlbench")
votesTib <- as_tibble(HouseVotes84)

map_dbl(votesTib, ~sum(is.na(.)))
map_dbl(votesTib, ~length(which(. == "y")))

votesUntidy <- gather(votesTib, "Variable", "Value", -Class)
votesUntidy %>% ggplot(aes(x = Class, fill = Value)) + facet_wrap(~Variable, scales = "free_y") + 
  geom_bar(position = "fill")

votesTask <- makeClassifTask(data = votesTib, target = "Class")
bayes <- makeLearner("classif.naiveBayes")
bayesModel <- train(learner = bayes, task = votesTask)

kfold <- makeResampleDesc("RepCV", folds = 10, reps = 50, stratify = TRUE)
bayesCV <- resample("classif.naiveBayes", task = votesTask, 
                    resampling = kfold,
                    measures = list(mmce, acc, fpr, fnr))

politician <- tibble(V1 = "n", V2 = "n", V3 = "y", V4 = "n", V5 = "n",
                     V6 = "y", V7 = "y", V8 = "y", V9 = "y", V10 = "y",
                     V11 = "n", V12 = "y", V13 = "n", V14 = "n",
                     V15 = "y", V16 = "n")
politicianPred <- predict(bayesModel, newdata = politician)
getPredictionResponse(politicianPred)


data(spam, package = "kernlab")
spamTib <- as_tibble(spam)
spamTib
spam.task
svm <- makeLearner("classif.svm")
getParamSet("classif.svm")
