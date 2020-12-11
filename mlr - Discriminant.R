#Discriminant 
library(mlr)
library(tidyverse)
install.packages("HDclassif")
data(wine, package = "HDclassif")
wineTib <- as_tibble(wine)
wineTib
names(wineTib) <- c("Class", "Alco", "Malic", "Ash", "Alk", "Mag",
                    "Phe", "Flav", "Non_flav", "Proan", "Col", "Hue",
                    "OD", "Prol")
wineTib$Class <- as.factor(wineTib$Class)
wineTib
wineUntidy <- gather(wineTib, key = "Variable", value = "Value", -Class)
wineUntidy %>% ggplot(aes(x = Class, y = Value)) + facet_wrap(~Variable, scales = "free_y") + 
  geom_boxplot() + theme_bw()

wineTask <- makeClassifTask(data = wineTib, target = "Class")
lda <-makeLearner("classif.lda")
ldamodel <- train(task = wineTask, learner = lda)
ldamodelData <- getLearnerModel(ldamodel)
ldapreds <- predict(ldamodelData)$x
head(ldapreds)

wineTib %>% 
  mutate(LD1 = ldapreds[,1],
         LD2 = ldapreds[,2]) %>% ggplot(aes(x = LD1, y = LD2, color = Class)) + 
  geom_point() + stat_ellipse() + theme_bw()

qda <- makeLearner("classif.qda")
qdamodel <- train(task = wineTask, learner = qda)

kfold <- makeResampleDesc("RepCV", folds = 10, reps = 50, stratify = TRUE)
ldaCV <- resample(learner = lda, task = wineTask, resampling = kfold, measures = list(mmce, acc))
qdaCV <- resample(learner = qda, task = wineTask, resampling = kfold, measures = list(mmce, acc))

calculateConfusionMatrix(ldaCV$pred, relative = TRUE)
calculateConfusionMatrix(qdaCV$pred, relative = TRUE)

poisoned <- tibble(Alco = 13, Malic = 2, Ash = 2.2, Alk = 19, Mag = 100,
                   Phe = 2.3, Flav = 2.5, Non_flav = 0.35, Proan = 1.7,
                   Col = 4, Hue = 1.1, OD = 3, Prol = 750)
predict(qdamodel, newdata = poisoned)


wineDiscr <- wineTib %>% mutate(LD1 = ldapreds[,1],
                                LD2 = ldapreds[,2]) %>%
  select(Class, LD1, LD2)
wineDiscrTask <- makeClassifTask(data = wineDiscr, target = "Class")
tunedk <- tuneParams(learner = "classif.knn",
                     task = wineDiscrTask, 
                     control = makeTuneControlGrid(),
                     par.set = makeParamSet(makeDiscreteParam("k", values = 1:10)),
                     resampling = makeResampleDesc("RepCV", folds = 10, reps = 20))

knntuningdata <- generateHyperParsEffectData(tunedk)
plotHyperParsEffect(knntuningdata, x = "k", y = "mmce.test.mean", plot.type = "line")

inner <- makeResampleDesc("CV")
outer <- makeResampleDesc("CV", iters = 10)
knnWrapper <- makeTuneWrapper("classif.knn", 
                              resampling = inner, 
                              par.set = makeParamSet(makeDiscreteParam("k", values = 1:10)),
                              control = makeTuneControlGrid())
cvwithtuning <- resample(knnWrapper, wineDiscrTask, resampling = outer)
tunedknn <- setHyperPars(makeLearner("classif.knn", par.vals = tunedk$x))
train(tunedknn, wineDiscrTask)
