### Predicting Solar Power #####
################################

######## ROMAIN JUBAN ##########
################################

#Load useful libraries
install.packages("ncdf4")
install.packages("gbm")
install.packages("lubridate")
install.packages("fossil")
library(ncdf4)
library(gbm)
library(lubridate)
library(fossil)

##Function to load the data

data_path="."

loadMesonetData <- function(filename, stationFilename="station_info.csv") {
  data=read.csv(file.path(data_path,filename))
  station_data=read.csv(file.path(data_path,stationFilename))
  list(data=data[,-1],dates=data[,1],station_data=station_data)
}

setwd("/afs/ir.stanford.edu/users/r/j/rjuban/Desktop/AMS")
listsubmit=loadMesonetData(file.path(data_path,"sampleSubmission.csv"))
dates=listsubmit$dates
header=c("Date",names(listsubmit$data))

#Training dataset : Daily incoming solar radiation for 98 Mesonet sites
trainingData=read.csv(file.path(data_path,"train.csv"))
trainingData$Date=strptime(trainingData$Date,"%Y%m%d")
stations=names(trainingData)[-1]

#Load weather parameters from GEFS grid weather stations for the training and test datasets
prefix_list = list("apcp_sfc","dlwrf_sfc","dswrf_sfc","pres_msl","pwat_eatm","spfh_2m","tcdc_eatm","tcolc_eatm","tmax_2m","tmin_2m","tmp_2m","tmp_sfc","ulwrf_sfc","ulwrf_tatm","uswrf_sfc")
suffix_train = "_latlon_subset_19940101_20071231.nc"
suffix_test = "_latlon_subset_20080101_20121130.nc"
parameter_names = list("Total_precipitation","Downward_Long-Wave_Rad_Flux","Downward_Short-Wave_Rad_Flux","Pressure","Precipitable_water","Specific_humidity_height_above_ground","Total_cloud_cover","Total_Column-Integrated_Condensate","Maximum_temperature","Minimum_temperature",  "Temperature_height_above_ground","Temperature_surface","Upward_Long-Wave_Rad_Flux_surface","Upward_Long-Wave_Rad_Flux","Upward_Short-Wave_Rad_Flux")

setwd("/afs/ir.stanford.edu/users/r/j/rjuban/AMS/train")
parameters=vector(mode="list",length=15)
for(i in 1:15){
  filename = paste0(prefix_list[[i]],suffix_train)
  file = nc_open(filename)
  parameters[[i]] = ncvar_get(file,parameter_names[[i]])
  if(i==1){
    intTime = ncvar_get(file,"intTime")
    lat=ncvar_get(file,'lat')
    lon=ncvar_get(file,'lon')-360
  }
  nc_close(file)
  print(i)
}

setwd("/afs/ir.stanford.edu/users/r/j/rjuban/Desktop/AMS/test")
parameters.test=vector(mode="list",length=15)
for(i in 1:15){
  filename = paste0(prefix_list[[i]],suffix_test)
  file = nc_open(filename)
  parameters.test[[i]] = ncvar_get(file,parameter_names[[i]])
  if(i==1){
    intTime.test = ncvar_get(file,"intTime")
  }
  nc_close(file)
  print(i)
}

#Average the weather data on 11 independent forecasted models to decrease the variance
parameters=lapply(parameters,function(x) colMeans(aperm(x,c(4,1,2,3,5))))
parameters.test=lapply(parameters.test,function(x) colMeans(aperm(x,c(4,1,2,3,5))))

#Formatting of dates
trainingDate=strptime(as.character(intTime/100),"%Y%m%d")
testingDate=strptime(as.character(intTime.test/100),"%Y%m%d")

#Altitude data for weather stations
filename="gefs_elevations.nc"
file=nc_open(filename)
elevation_control=ncvar_get(file,"elevation_control")

#Month data (featurization of the date)
month_feature=rep(NA,5113)
for (i in 1:5113){
  month_feature[i]=month(ymd(as.character(trainingData$Date[i])))
}

month_test_feature=rep(NA,1796)
for (i in 1:1796){
  month_test_feature[i]=month(ymd(as.character(listsubmit$dates[i])))
}

#Main function : Train a model on the training dataset and make the prediction of daily incoming solar energy
#on the test data set
outdata = lapply(stations, function(station){
  cat(station,"\n")
  
  #(lon,lat) position of the Mesonet station
  nlat=listsubmit$station_data[listsubmit$station_data$stid==station,2]
  elon=listsubmit$station_data[listsubmit$station_data$stid==station,3]
  
  #Harvesine distance to the four closest GEFS grid weather stations
  distance1=matrix(rep(deg.dist(elon,nlat,floor(elon),floor(nlat)),5113), ncol = 1)
  distance2=matrix(rep(deg.dist(elon,nlat,floor(elon),ceiling(nlat)),5113), ncol = 1)
  distance3=matrix(rep(deg.dist(elon,nlat,ceiling(elon),floor(nlat)),5113), ncol = 1)
  distance4=matrix(rep(deg.dist(elon,nlat,ceiling(elon),ceiling(nlat)),5113), ncol = 1)
  
  #Altitude of the four closest GEFS grid weather stations
  altitude1=matrix(rep(elevation_control[match(floor(elon),lon),match(floor(nlat),lat)],5113), ncol = 1)
  altitude2=matrix(rep(elevation_control[match(floor(elon),lon),match(ceiling(nlat),lat)],5113), ncol = 1)
  altitude3=matrix(rep(elevation_control[match(ceiling(elon),lon),match(floor(nlat),lat)],5113), ncol = 1)
  altitude4=matrix(rep(elevation_control[match(ceiling(elon),lon),match(ceiling(nlat),lat)],5113), ncol = 1)
  
  #Weather parameters for the four closest GEFS grid weather stations (training)
  parameters_station1=do.call("rbind",lapply(parameters,function(x) x[match(floor(elon),lon),match(floor(nlat),lat),,]))
  parameters_station2=do.call("rbind",lapply(parameters,function(x) x[match(floor(elon),lon),match(ceiling(nlat),lat),,]))
  parameters_station3=do.call("rbind",lapply(parameters,function(x) x[match(ceiling(elon),lon),match(floor(nlat),lat),,]))
  parameters_station4=do.call("rbind",lapply(parameters,function(x) x[match(ceiling(elon),lon),match(ceiling(nlat),lat),,]))
  
  #Construction of the training dataset for each Mesonet site
  training_features=t(rbind(parameters_station1,parameters_station2,parameters_station3,parameters_station4))
  new_features=cbind(distance1,distance2,distance3,distance4,altitude1,altitude2,altitude3,altitude4,month_feature)
  training_features=cbind(training_features,new_features)
  
  ##Gradient Boosting Model##
  model=gbm(trainingData[,station]~.,data=data.frame(training_features),distribution="gaussian",interaction.depth=5,n.trees=1000,shrinkage=0.001,cv.folds=10)
  best.iter=gbm.perf(model,method="cv")
  ##Gradient Boosting Model##
  
  #Weather parameters for the four closest GEFS grid weather stations (testing)
  parameters.test_station1=do.call("rbind",lapply(parameters.test,function(x) x[match(floor(elon),lon),match(floor(nlat),lat),,]))
  parameters.test_station2=do.call("rbind",lapply(parameters.test,function(x) x[match(floor(elon),lon),match(ceiling(nlat),lat),,]))
  parameters.test_station3=do.call("rbind",lapply(parameters.test,function(x) x[match(ceiling(elon),lon),match(floor(nlat),lat),,]))
  parameters.test_station4=do.call("rbind",lapply(parameters.test,function(x) x[match(ceiling(elon),lon),match(ceiling(nlat),lat),,]))
  
  #Construction of the test dataset for each Mesonet site
  test_features=t(rbind(parameters.test_station1,parameters.test_station2,parameters.test_station3,parameters.test_station4))
  new_test_features=new_features[,-9]
  new_test_features=new_test_features[1:1796,]
  new_test_features=cbind(new_test_features,month_test_feature)
  test_features=cbind(test_features,new_test_features)
  
  ##Prediction##
  preds=predict(model,data.frame(test_features),best.iter) 
  preds=preds*(preds>=0)
  ##Prediction##
})

setwd("/afs/ir.stanford.edu/users/r/j/rjuban/Desktop/AMS")

#Formatting for the output submission file
dates = as.character(intTime.test/100)
df = data.frame(matrix(unlist(outdata),nrow=length(dates),byrow=F))
df=cbind(dates,df)
colnames(df)=c("Date",stations)
write.csv(df,"submission.csv",row.names=FALSE)