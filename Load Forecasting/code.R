###### Load Forecasting ########
################################

######## ROMAIN JUBAN ##########
################################



#Loading packages

setwd("~/Desktop/Load Forecasting")
install.packages("reshape")
library(reshape)
install.packages("stringr")
library(stringr)
install.packages("glmnet")
library(glmnet)
install.packages("forecast")
library(forecast)
install.packages("randomForest")
library(randomForest)
install.packages("gbm")
library(gbm)


#Loading data
load.history = read.csv("load_history.csv",header=TRUE,as.is=TRUE)
temperature.history = read.csv("temperature_history.csv",header=TRUE,as.is=TRUE)
submission.template=read.csv("submission_template.csv",header=TRUE)
weights=read.csv("weights.csv",header=TRUE)
test=read.csv("test.csv",header=TRUE)

#Reshape dataframes (melting, date formatting and cleaning)
data.shape <- function(df){
  Mdf <- melt(df,id=1:4)
  Mdf$hour <- substr(Mdf$variable,start=2,stop=3)
  Mdf$date <- ISOdate(year=Mdf$year,month=Mdf$month,day=Mdf$day,hour=Mdf$hour)
  data <- Mdf[,c(1,8,6)]
  data <- data[order(data[,1],data[,2]),]
  row.names(data) <- NULL
  data[,1] <- factor(data[,1])
  data
}

#Data frames for Load and Temperature data
data.load <- data.shape(load.history)
data.load[,3] <- as.numeric(gsub(",","", data.load[,3]))
colnames(data.load)[3] <- "load"
data.temperature <- data.shape(temperature.history)
data.temperature <- reshape(data.temperature,direction="wide",timevar="station_id",idvar=c("date"))

#Missing temperature data solution
temperature.solution = read.csv('temperature_solution.csv',header=TRUE,as.is=TRUE)
temperature.solution$date <- ISOdate(year=temperature.solution$year,month=temperature.solution$month,day=temperature.solution$day,hour=temperature.solution$hour)
temperature.solution <- temperature.solution[,c(1,3,8)]
temperature.solution <- reshape(temperature.solution,direction="wide",timevar="station_id",idvar=c("date"))
data.temperature <- data.temperature[(data.temperature$date < as.POSIXct("2008-06-30 07:00:00",tz="GMT")),]
data.temperature <- rbind(data.temperature,setNames(temperature.solution,names(data.temperature)))

#Merging of load and temperature data (outer join)
data <- merge(data.load,data.temperature,by="date",all=TRUE,sort=FALSE)
data <- data[order(data[,2],data[,1]),]
row.names(data) <- NULL
colnames(data)[4:14] <- paste("T",1:11,sep="")

#Creation of additional variables
#Trend (numeric)
#Month, Hour, Weekday (Factors)
data$load <- as.numeric(data$load)
data$trend <- 1:nrow(data)
data$date <- as.POSIXct(data$date)
data$month <- as.factor(format(data$date,format="%m"))
data$hour <- as.factor(format(data$date,format="%H"))
data$weekday <- as.factor(weekdays(data$date))

#Jump of zone 10
#jump = mean(data$load[format(data$date,"%Y")>=2008 & data$zone_id == 10],na.rm=TRUE)-mean(data$load[format(data$date,"%Y")<2008 & data$zone_id == 10],na.rm=TRUE)
#data$load[format(data$date,"%Y")>=2008 & data$zone_id == 10] = data$load[format(data$date,"%Y")>=2008 & data$zone_id == 10] - jump

#Weights for Exponentially weighted least squares
weight <- 1.00015
data$weights <- weight^seq(from=0,to=nrow(data)-1,by=1)

#Splitting of the data into 20 different zones
s.data <- split(data,data$zone_id)

#Interpolation of temperature for the last week of prediction (time series approach)
# data <- do.call(rbind,lapply(1:20,function(i){
#   cat("station",i,"\n")
#   df <- s.data[[i]]
#   df.train <- df[df$date >= as.POSIXct("2008-01-01 00:00:00",tz="GMT") & 
#                   df$date < as.POSIXct("2008-06-30 07:00:00",tz="GMT"),]
#   df.test <- df[(df$date >= as.POSIXct("2008-06-30 07:00:00",tz="GMT")),]
#   for (j in 1:11){
#     temp.train <- df.train[,c(3+j)]
#     temp.ts <- ts(temp.train,frequency=24)
#     df.test[,c(3+j)] <- round(stlf(temp.ts,h=nrow(df.test),s.window="periodic",method="arima")$mean)
#   }
#   rbind(df[(df$date < as.POSIXct("2008-01-01 00:00:00",tz="GMT")),],df.train,df.test)
# }))

#Splitting of the data into training and test set by zone
data.train <- data[!is.na(data$load),]
data.test <- data[is.na(data$load),]
s.train <- split(data.train,data.train$zone_id)
s.test <- split(data.test,data.test$zone_id)

#Cross-validation to select the best temperature station for each zone id
selected_temp <- as.numeric(lapply(1:20,function(i){
  cat("station",i,"\n")
  df <- s.train[[i]]
  rmse <- lapply(1:11,function(x){
    names(df)[names(df) == paste("T",x,sep="")] <- "temp"
    df.train <- df[df$date < as.POSIXct("2007-01-01 00:00:00",tz="GMT"),]
    df.test <- df[df$date >= as.POSIXct("2007-01-01 00:00:00",tz="GMT"),]
    model <- lm(load~trend+weekday+hour+month+month:temp+month:temp+month:I(temp^2)+
                  month:I(temp^3)+hour:temp+hour:I(temp^2)+hour:I(temp^3),data=df.train)
    preds <- predict(model,newdata=df.test)
    sqrt(mean((preds-df.test$load)^2))
  })
  which.min(as.numeric(rmse))
}))

#Result : selected_temp
#[1]  2  9  9  2  9  9  9 11  6  5  5  5  2  4  6  7 10  9 10 11
selected_temp <- c(2,9,9,2,9,9,9,11,6,5,5,5,2,4,6,7,10,9,10,11)

#Backcasting hourly load (8 weeks) and forecasting hourly load (1 week)
s.preds <- lapply(1:20,function(i){
  cat("station",i,"\n")
  df.train <- s.train[[i]][,c(3,3+selected_temp[i],15,16,17,18,19)]
  df.test <- s.test[[i]][,c(3+selected_temp[i],15,16,17,18,19)]
  names(df.train)[2] <- "temp"
  names(df.test)[1] <- "temp"
  
  f <- ~trend+weekday+hour+month+month:temp+month:temp+month:I(temp^2)+month:I(temp^3)+hour:temp+hour:I(temp^2)+hour:I(temp^3)
  x.train <- model.matrix(f,df.train)
  x.test <- model.matrix(f,df.test)
  y.train <- as.matrix(df.train$load,ncol=1)
  
  ##Models
  #Linear regression
  model <- lm(load~trend+weekday+hour+month+month:temp+month:temp+month:I(temp^2)+
  month:I(temp^3)+hour:temp+hour:I(temp^2)+hour:I(temp^3),data=df.train,weights=df.train[,7])
  preds <- predict(model,newdata=df.test,weights=df.test[,6])
  preds
})

preds <- unlist(s.preds)
data.preds <- data.frame(data.test[,c(1,2,16,17)],preds)
data.preds <- data.preds[!(data.preds$date < as.POSIXct("2008-07-01 01:00:00",tz="GMT") & 
                             data.preds$date > as.POSIXct("2008-06-30 06:00:00",tz="GMT")),]

#Jump of Zone 10, correction
#data.preds$preds[format(data.preds$date,"%Y")>=2008 & data.preds$zone_id == 10] = data.preds$preds[format(data.preds$date,"%Y")>=2008 & data.preds$zone_id == 10] + jump

#Forecasting at the utility level (aggregate of the 20 zones)
dates <- unique(data.preds$date)
for (i in 1:length(dates)) {
  d <- dates[i]
  system.preds <- sum(data.preds$preds[data.preds$date == d])
  newr <- data.frame(date=d,zone_id="21",month=format(d,format="%m"),
                     hour=format(d,format="%H"),preds=system.preds)
  data.preds <- rbind(data.preds,newr)
}

#Prediction data formatting
data.preds$date <- as.Date(data.preds$date)
levels(data.preds$hour)[1] <- "24"
data.preds$date[data.preds$hour == "24"] <- as.Date(data.preds$date[data.preds$hour == "24"])-1
data.preds$day <- as.factor(format(data.preds$date,format="%d"))
data.preds$month <- as.factor(format(data.preds$date,format="%m"))
data.preds$year <- as.factor(format(data.preds$date,format="%Y"))
data.preds$preds <- round(data.preds$preds)
md.preds <- reshape(data.preds,direction="wide",timevar="hour",idvar=c("date","year","month","day","zone_id"))
md.preds <- md.preds[order(md.preds[,1],md.preds[,2]),]
submission.template[,6:29] <- md.preds[,6:29]


#Comparison with the solution data
load.solution <- read.csv("load_solution.csv",header=TRUE)
error <- apply(weights[,6:29]*(load.solution[,6:29]-submission.template[,6:29])^2,1,mean)
error <- sqrt(sum(error)/sum(load.solution[,30]))
error


#Methods comparison
#Linear Regression lambda=1 error=86848.64
#Linear Weighted Regression lambda=1.00015  error=84568.52
#Lasso (lambda chosen by CV) error=86848.64
#Ridge (lambda chosen by CV) error=86848.64
#Random Forest mtry=2, ntree=50 error=79176.24
#Random Forest mtry=2, ntree=100 error=78915.94
#Random Forest mtry=3, ntree=200 error=79146.22
