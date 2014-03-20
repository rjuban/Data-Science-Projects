#### Predicting Temperature ####
################################

######## ROMAIN JUBAN ##########
################################

#Packages used
install.packages("forecast")
library(forecast)

install.packages("kknn")
library(kknn)

install.packages("gam")
library(gam)

#Part I: Predicting temperatures in 2011 at 500 different sites from the training set
#Loading Files
training <- read.csv("training.csv",header=TRUE,as.is=TRUE)
print("Part 2: Loading file...")
locations.ids <- names(training)[-1]
start.date <- training[1,1]
end.date <- training[nrow(training),1]

#Remove Feb 29's so that each year has 365 days
training <- training[format(as.POSIXct(training[,1]),"%m%d") != "0229",]
time_index <- as.POSIXct(training[,1],"%Y%m%d")
new.time_index <- seq(from = as.Date("2011-01-01"),to = as.Date("2011-12-31"),by="day")

#Some plots to understand the data
#Time series of temperatures for a given location
train.ts <- ts(training[,2],frequency=365,start=c(1980,1))
plot(train.ts,type='l',ylab="Temperature",main="Time series of temperatures for a given location")

#Seasonal decomposition by moving averages
plot(decompose(train.ts))

#Seasonal plot
seasonplot(train.ts,col=rainbow(12),type='l')

#Matrix of dummy variables to factor the month as a regressor variable in the model
month <- as.factor(format(time_index,"%m"))
new.month <- as.factor(format(new.time_index,"%m"))
xreg <- model.matrix(~as.factor(month))[,2:12]
newxreg <- model.matrix(~as.factor(new.month))[,2:12]

#Test of the model on 1 location
#STL model (Seasonal Trend Decomposition) with month factored as a regressor variable
fit <- stlf(train.ts,h=365,s.window="periodic",method="arima",xreg=xreg,newxreg=newxreg)

#Plot of the prediction
plot.forecast(fit,xlim=c(2005,2012),xlab="Dates",ylab="Temperature",main="Prediction of the temperature in 2011")

#Prediction of temperatures for 2011 (365 days) at 500 different locations
forecasts <- matrix(0,365,500)
for (i in 1:500){
  cat(i," ")
  train.ts <- ts(training[,i+1],frequency=365,start=c(1980,1))
  train.stlf <- stlf(train.ts,h=365,s.window="periodic",method="arima",xreg=xreg,newxreg=newxreg)
  forecasts[,i] <- train.stlf$mean
}
print("Prediction for 2011 at 500 different sites done.")

#Submission
forecasts <- data.frame(new.time_index,round(forecasts[-1]))
colnames(forecasts) <- colnames(training)
write.csv(forecasts,"forecasts.csv",row.names=FALSE)
print("Submission for part 1 done.")

##Part 2: Predict temperatures in 2011 at 50 different new locations 
#Loading files
print("Part 2: Loading file...")
locations <- read.csv("locations.csv",header=TRUE,as.is=TRUE)
old.locations <- locations[1:500,]
new.locations <- locations[501:550,]

#Plot of old and new locations on a (lonxlat) grid
plot(old.locations[,5],old.locations[,4],pch=18,col="blue",xlab="Longitude",ylab="Latitude",main="Locations of sites")
points(new.locations[,5],new.locations[,4],pch=20,col="red")

#K-Nearest Neighbours regression to find the temperature forecasts at the new stations
#K=2 chosen by cross-validation
new.forecasts=matrix(0,365,50)
for (i in 1:365){
  cat(i," ")
  data <- data.frame(temp=as.numeric(forecasts[i,-1]),old.locations[,4:5])
  knn.reg <- kknn(temp~.,data,new.locations[,4:5],k=2,distance=1,kernel="optimal")
  new.forecasts[i,] <- knn.reg$fitted.values
}
print("Interpolation for the 50 new sites done.")

##Submission
new.forecasts <- data.frame(new.time_index,round(new.forecasts))
colnames(new.forecasts) <- c("dates",locations[501:550,1])
write.csv(new.forecasts,"new.forecasts.csv",row.names=FALSE)
print("Submission for part 2 done.")

###Part 3: Validation of the models
#We will test the models on a sample of 10 sites randomly selected
set.seed(1)
train.idx <- sample(2:501,10)

#Validation of the size of the training set
#The entire training set gives the lowest error
starting.year <- 1980:2007
errors <- matrix(0,length(train.idx),length(starting.year))
for (i in 1:length(train.idx)){
  cat(i," ")
  for (j in 1:length(starting.year)){
    training.ts <- ts(training[,train.idx[i]],frequency=365,start=c(starting.year[j],1))
    train.ts <- window(training.ts,start=c(starting.year[j],1),end=c(2009,365))
    test.ts <- window(training.ts,start=c(2010,1),end=c(2010,365))
    fit.stlf <- stlf(train.ts,h=365,s.window="periodic",method="arima")
    errors[i,j] <- mean((fit.stlf$mean-test.ts)^2)
  } 
}
output <- colMeans(errors)
plot(output,type='l')

#Selection of the forecasting method for forecasting temperature at the 500 sites
#Best model: STL (Seasonal Trend Decomposition) with month factored as a regressor variable
#We split the training data (1980 -> 2010) in a training set (1980 -> 2009) and a validation set (2010)

time_index <- seq(from = as.Date("1980-01-01"),to = as.Date("2009-12-31"),by="day")
time_index <- time_index[format(time_index,"%m%d") != "0229"]
new.time_index <- seq(from = as.Date("2010-01-01"),to = as.Date("2010-12-31"),by="day")
new.time_index <- new.time_index[format(new.time_index,"%m%d") != "0229"]

month <- as.factor(format(time_index,"%m"))
new.month <- as.factor(format(new.time_index,"%m"))
xreg <- model.matrix(~as.factor(month))[,2:12]
newxreg <- model.matrix(~as.factor(new.month))[,2:12]

errors <- matrix(0,length(train.idx),4)
for (i in 1:length(train.idx)){
  cat(i," ")
  training.ts <- ts(training[,train.idx[i]],frequency=365,start=c(1980,1))
  train.ts <- window(training.ts,start=c(1980,1),end=c(2009,365))
  test.ts <- window(training.ts,start=c(2010,1),end=c(2010,365))
  fit.stlf <- stlf(train.ts,h=365,s.window="periodic",method="arima")
  fit.stlf.month <- stlf(train.ts,h=365,s.window="periodic",method="arima",xreg=xreg,newxreg=newxreg)
  fit.tslm <- tslm(train.ts~trend+season)
  fc.tslm <- forecast(fit.tslm,h=365)
  fit.hw <- HoltWinters(train.ts)
  fc.hw <- forecast.HoltWinters(fit.hw,h=365)
  preds <- list(fit.stlf$mean,fit.stlf.month$mean,fc.tslm$mean,fc.hw$mean)
  errors[i,] <- unlist(lapply(preds,function(x){mean((x-test.ts)^2)}))
}
output <- colMeans(errors)
plot(output)

#Validation of the interpolation method for the 50 new sites
#Best method:Weighted K-Nearest Neighbor with K=2
#We split the 500 old sites in a training (400) and validation set (100)
set.seed(1)
train <- sample(1:500,400)
test <- -train
train.locations <- old.locations[train,]
test.locations <- old.locations[test,]

#Date column removed
temp.train <- training[,-1]

plot(train.locations[,5],train.locations[,4],pch=18,col="blue")
points(test.locations[,5],test.locations[,4],pch=20,col="red")

#Weighted K-Nearest Neighbor using Minkowski distance
#K=2 chosen by cross-validation
output <- rep(0,10)
for (deg in 1:10){
  cat(deg," ")
  errors <- rep(0,365)
  for (i in 1:365){
    data <- data.frame(temp=as.numeric(temp.train[i,train]),train.locations[,4:5])
    knn.reg <- kknn(temp~.,data,test.locations[,4:5],k=deg,distance=1,kernel="optimal")
    errors[i] <- mean((temp.train[i,test]-knn.reg$fitted.values)^2)
  }
  output[deg] <- mean(errors)
}
plot(output,type='l')

#Spline interpolation with a GAM model
#Degrees of freedom chosen:15
output <- rep(0,20)
for (deg in 1:20){
  cat(deg," ")
  errors <- rep(0,365)
  for (i in 1:365){
    data=data.frame(temp=as.numeric(temp.train[i,train]),train.locations[,4:5])
    fit <- gam(temp~s(lat,deg)+s(long,deg),data=data)
    preds <- predict(fit,newdata=test.locations[,4:5])
    errors[i] <- mean((temp.train[i,test]-preds)^2)
  }
  output[deg] <- mean(errors)
}
plot(output,type='l')



############END OF CODE#############


##Visualization

install.packages("ggplot2")
library(ggplot2)

install.packages("RColorBrewer")
library(RColorBrewer)

#Comparison of yearly averaged temperature
avg.forecasts <- colMeans(forecasts[,-1])
avg.forecasts <- cut(avg.forecasts,9)
data <- data.frame(locations[1:500,4:5],avg.forecasts)
colours <- brewer.pal(name="RdYlGn",n=9)
colours <- rev(colours)
p <- ggplot(data,aes(x=long, y=lat))+geom_point(aes(size=2,colour = avg.forecasts))+scale_colour_manual(values=colours)
p <- p+coord_fixed(ratio=0.9)+labs(x="Longitude",y="Latitude",title="Average temperature in 2011")+scale_x_continuous(breaks=-104:-94)+scale_y_continuous(breaks=37:43)
p

avg.observed <- colMeans(training[10951:11315,-1])
avg.observed <- cut(avg.observed,9)
data <- data.frame(locations[1:500,4:5],avg.observed)
colours <- brewer.pal(name="RdYlGn",n=9)
colours <- rev(colours)
p <- ggplot(data,aes(x=long, y=lat))+geom_point(aes(size=2,colour = avg.observed))+scale_colour_manual(values=colours)
p <- p+coord_fixed(ratio=0.9)+labs(x="Longitude",y="Latitude",title="Average temperature in 2010")+scale_x_continuous(breaks=-104:-94)+scale_y_continuous(breaks=37:43)
p

#Comparison of maximal temperature
#Function colMax similar to colMeans
colMax <- function(X){
  apply(X,2,max)
}
max.forecasts <- colMax(forecasts[,-1])
max.forecasts <- cut(max.forecasts,9)
data = data.frame(locations[1:500,4:5],max.forecasts)
colours=brewer.pal(name="YlOrRd",n=9)

p=ggplot(data,aes(x=long, y=lat))+geom_point(aes(size=2,colour = max.forecasts))+scale_colour_manual(values=colours)
p=p+coord_fixed(ratio=0.9)+labs(x="Longitude",y="Latitude",title="Maximum temperature in 2011")+scale_x_continuous(breaks=-104:-94)+scale_y_continuous(breaks=37:43)
p

max.observed <- colMax(training[10951:11315,-1])
max.observed <- cut(max.observed,9)
data = data.frame(locations[1:500,4:5],max.observed)
colours=brewer.pal(name="YlOrRd",n=9)

p <- ggplot(data,aes(x=long, y=lat))+geom_point(aes(size=2,colour = max.observed))+scale_colour_manual(values=colours)
p <- p+coord_fixed(ratio=0.9)+labs(x="Longitude",y="Latitude",title="Maximum temperature in 2010")+scale_x_continuous(breaks=-104:-94)+scale_y_continuous(breaks=37:43)
p

#Comparison of minimmal temperature
#Function colMin similar to colMeans
colMin <- function(X){
  apply(X,2,min)
}
min.forecasts <- colMin(forecasts[,-1])
min.forecasts <- cut(min.forecasts,9)
data <- data.frame(locations[1:500,4:5],min.forecasts)
colours <- brewer.pal(name="Blues",n=9)
colours <- rev(colours)
p <- ggplot(data,aes(x=long, y=lat))+geom_point(aes(size=2,colour = min.forecasts))+scale_colour_manual(values=colours)
p <- p+coord_fixed(ratio=0.9)+labs(x="Longitude",y="Latitude",title="Minimum temperature in 2011")+scale_x_continuous(breaks=-104:-94)+scale_y_continuous(breaks=37:43)
p

min.observed <- colMin(training[10951:11315,-1])
min.observed <- cut(min.observed,9)
data <- data.frame(locations[1:500,4:5],min.observed)
colours <- brewer.pal(name="Blues",n=9)
colours <- rev(colours)
p <- ggplot(data,aes(x=long, y=lat))+geom_point(aes(size=2,colour = min.observed))+scale_colour_manual(values=colours)
p <- p+coord_fixed(ratio=0.9)+labs(x="Longitude",y="Latitude",title="Minimum temperature in 2010")+scale_x_continuous(breaks=-104:-94)+scale_y_continuous(breaks=37:43)
p

#Monthly insolation
data <- data.frame(training[,1:2])
data$month <- as.numeric(format(as.yearmon(training[,1]),"%m"))
data <- data[,-1]
data <- aggregate(location_1~month,data,mean)
p <- ggplot(data,aes(x=month,y=location_1))+geom_bar(stat="identity",aes(fill=location_1))+scale_fill_gradient(low="yellow",high="red")
p <- p+labs(x="Month",y="Temperature",title="Average monthly temperature")
p
