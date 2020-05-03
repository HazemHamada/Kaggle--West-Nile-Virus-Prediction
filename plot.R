setwd("D:/Engineering/Additional Projects/Pending/Kaggle- West Nile Virus Prediction")

plotBlock = function(plotYear) {
  require(ggplot2)
  require(plyr)
  
  dataFolder = "D:/Engineering/Additional Projects/Pending/Kaggle- West Nile Virus Prediction"
  dfSpray = read.csv(file.path(dataFolder,"spray.csv"))
  dfTrain = read.csv(file.path(dataFolder,"train.csv"))
  dfTrain$year = as.POSIXlt(dfTrain$Date)$year + 1900
  dfTrain$woy = as.POSIXlt(dfTrain$Date)$yday / 7
  dfSpray$year = as.POSIXlt(dfSpray$Date)$year + 1900
  dfTrain = subset(dfTrain,dfTrain$year %in% plotYear)
  dfSpray = subset(dfSpray,dfSpray$year %in% plotYear)
  # merge records for same trap, same week
  dfBlock = ddply(dfTrain,.(Block,woy,year,Longitude,Latitude),summarize,
                  NumMosquitos = sum(NumMosquitos),
                  WnvPresent = ifelse(is.na(match(1,WnvPresent)),"no","yes"))
  blockPlot = ggplot(dfBlock) + 
    coord_cartesian(xlim=c(-87.97,-87.5),ylim=c(41.62,42.06)) +
    geom_point(aes(x=Longitude,y=Latitude), data=dfSpray,size=4, colour="#ffcaca",alpha=1) +
    geom_point(aes(x=Longitude,y=Latitude, size=NumMosquitos, colour=WnvPresent),shape=1) +
    scale_size(range=c(4,50)) + 
    guides(size=FALSE) +
    guides(colour=guide_legend(title="West Nile Virus Present", override.aes=list(size=4))) +
    scale_color_manual(name="",values=c("black","red")) + 
    theme(panel.background=element_rect(fill="grey95"), panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(), legend.position="bottom") +
    ggtitle("West Nile Virus Detection by Block")
  if (length(plotYear > 1)) {
    
    blockPlot = blockPlot + facet_wrap(~year,ncol=2)
  }
  fname = paste("BlockPlot",paste(plotYear,collapse="_"),".png",sep="_")
  png(fname,width=7,height=10,units="in",res=72)
  print(blockPlot)
  dev.off()
}
plotBlock(c(2007,2009,2011,2013))
plotBlock(2013)