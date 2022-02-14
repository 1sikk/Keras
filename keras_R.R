install.packages("mlbench")
library(mlbench)
install.packages("car")
library(car)
data("BostonHousing")

formula <- medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + 
  tax + ptratio + b + lstat
lm_boston <- lm(formula, data = BostonHousing)
lm2_boston <- step(lm_boston, direction="both")
# 최적의 포뮬러  
# AIC가 낮을수록 좋은 결과치이다.
# AIC 아카이케정보기준. 품질을 나타냄
formula2 <- medv ~ crim + zn + chas + nox + rm + dis + rad + tax + ptratio + 
  b + lstat

lm3_boston <- lm(formula2, data = BostonHousing)
lm3_boston
summary(lm3_boston)

# 트레이닝, 테스트 셋으로 나누기
idx = sample(1:nrow(BostonHousing), 0.7*nrow(BostonHousing))
idx             
train <- BostonHousing[idx,]
test <-  BostonHousing[-idx,]

# 트레이닝 셋으로 예측하기
fit.lm <- lm(formula, data = train)
summary(fit.lm)
# 예측하기
pred.lm <- predict(fit.lm, test)
cor(pred.lm,test$medv)
# 예측치 83 % 