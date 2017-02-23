# Georgetown University
# SEST 704 Intelligence Analytics
# Lab 3: Computer Exercises
# Section 5: RStudio
# 22 February 2017

# References
# [1] http://www.ats.ucla.edu/stat/r/dae/logit.htm

# 5.1. Initial data exploration, reproducing results from Excel

# Clear variables, close all plots, restore defaults
rm(list=ls())
dev.off()

# Install library to plot
install.packages("ggplot2")
install.packages("RColorBrewer")

library(ggplot2)
library(RColorBrewer)               #for brewer.pal()

# Read data
mydata<-read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")

# Make quick histograms
hist(mydata$gre,main="GRE Histogram",xlab = "GRE score", ylab = "Frequency")
hist(mydata$gpa,main="GPA Histogram",xlab = "GPA", ylab = "Frequency")
hist(mydata$rank,main="Prestige Histogram",xlab = "Prestige", ylab = "Frequency")

# Let's try that again, without the defaults--should be closer to Excel results
# Two ways of specifying the bin centers: c and seq
hist(mydata$gre,main="GRE Histogram",xlab = "GRE score", ylab = "Frequency",breaks=c(50,150,250,350,450,550,650,750,850))
hist(mydata$gpa,main="GPA Histogram",xlab = "GPA", ylab = "Frequency",breaks=seq(2.125,4.125,0.25))
hist(mydata$rank,main="Prestige Histogram",xlab = "Prestige", ylab = "Frequency",breaks=c(0.5,1.5,2.5,3.5,4.5))

# 5.2 Do logistic regression to see how important each factor is
mydata$rank <- factor(mydata$rank)
mylogit <- glm(admit ~ gre + gpa + rank, data = mydata, family = "binomial")
#print(summary(mylogit))

## 5.2 Logistic regressions. Develop intuition by point, by line, then by area

# 5.2.1 Prediction for single data point
newdata = data.frame(gre = 690, gpa = 3.859, rank = factor(2))
predict(mylogit,newdata,type="response")

# 5.2.2 Prediction for curve by varying GPA and rank, but keeping GRE constant
newdata_constant_GRE <- with(mydata,data.frame(gre = mean(gre),gpa = rep(seq(from = 2.25, to = 4,length.out=100), 4), rank = factor(rep(1:4, each = 100))))
print(newdata_constant_GRE)
newdata_constant_GRE$Prob <- predict(mylogit, newdata = newdata_constant_GRE, type = "response")
ggplot(newdata_constant_GRE, aes(x = gpa, y = Prob))+ 
  geom_line(aes(colour = rank))

# Prediction for curve by varying GRE and rank, but keeping GPA constant
newdata_constant_GPA <- with(mydata,data.frame(gre = rep(seq(from = 200, to = 800, length.out = 100), 4),gpa = mean(gpa), rank = factor(rep(1:4, each = 100))))
print(newdata_constant_GPA)
newdata_constant_GPA$Prob <- predict(mylogit, newdata = newdata_constant_GPA, type = "response")
ggplot(newdata_constant_GPA, aes(x = gre, y = Prob))+ 
  geom_line(aes(colour = rank))


# 5.2.3  Make filled, rainbow contour plot across varying GRE and GPA with prestige constant
xgrid <-  seq(200, 800, 10) # GRE
ygrid <-  seq(2,4, 0.05)    # GPA
xygrid <-  expand.grid(gre = xgrid, gpa = ygrid)
nx = length(xgrid)
ny = length(ygrid)
contour_data <-cbind(xygrid,rank=factor(1)) # Throw in prestige
predicted_P <-  matrix(predict(mylogit, contour_data,type="response"),nx,ny)
c1<-filled.contour(x=xgrid,
               y=ygrid,
               z=predicted_P,
               zlim = range(seq(0, 0.8, by = 0.10), finite=TRUE),
               levels = seq(0, 0.8, by = 0.01),
               color.palette = rainbow,
               col=rainbow(128),
               #              col = heat.colors(100, alpha = 1),
               plot.title = title(main = "Probability of Admission with Prestige = 1",xlab="GRE",ylab="GPA"),
               labels=c(seq(0,.8, by=0.1),rep("", 16) ),
               key.title = title(main="P")
)

# Make minimalist labeled countour plot varying GRE and GPA, keeping prestige constant
x=xgrid
y=ygrid
z=predicted_P
d3<-expand.grid(x=xgrid,y=ygrid)
d3$z<-as.vector(predicted_P)
contourplot(z~x+y,data=d3,xlab = "GRE",ylab="GPA",main="Probability of Admission with Prestige = 1")
