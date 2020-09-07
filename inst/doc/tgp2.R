### R code from vignette source 'tgp2.Rnw'

###################################################
### code chunk number 1: tgp2.Rnw:33-35
###################################################
library(tgp)
options(width=65)


###################################################
### code chunk number 2: tgp2.Rnw:148-149
###################################################
seed <- 1; set.seed(seed)  ## seed zero problematic with null btlm map tree below


###################################################
### code chunk number 3: tgp2.Rnw:214-218
###################################################
fb.train <- fried.bool(500)
X <- fb.train[,1:13]; Z <- fb.train$Y
fb.test <- fried.bool(1000)
XX <- fb.test[,1:13]; ZZ <- fb.test$Ytrue


###################################################
### code chunk number 4: tgp2.Rnw:224-225
###################################################
names(X)


###################################################
### code chunk number 5: tgp2.Rnw:232-235
###################################################
fit1 <- bcart(X=X, Z=Z, XX=XX, verb=0)
rmse1 <- sqrt(mean((fit1$ZZ.mean - ZZ)^2))
rmse1


###################################################
### code chunk number 6: cat-fbcart-mapt
###################################################
tgp.trees(fit1, "map")


###################################################
### code chunk number 7: tgp2.Rnw:244-245
###################################################
graphics.off()


###################################################
### code chunk number 8: tgp2.Rnw:259-262
###################################################
fit2 <- btlm(X=X, Z=Z, XX=XX, verb=0)
rmse2 <- sqrt(mean((fit2$ZZ.mean - ZZ)^2))
rmse2


###################################################
### code chunk number 9: cat-fbtlm-mapt
###################################################
tgp.trees(fit2, "map")


###################################################
### code chunk number 10: tgp2.Rnw:270-271
###################################################
graphics.off()


###################################################
### code chunk number 11: tgp2.Rnw:305-308
###################################################
fit3 <- btlm(X=X, Z=Z, XX=XX, basemax=10, verb=0)
rmse3 <- sqrt(mean((fit3$ZZ.mean - ZZ)^2))
rmse3


###################################################
### code chunk number 12: cat-fbtlm-mapt
###################################################
tgp.trees(fit3, "map")


###################################################
### code chunk number 13: tgp2.Rnw:314-315
###################################################
graphics.off()


###################################################
### code chunk number 14: tgp2.Rnw:337-340
###################################################
fit4 <- btgpllm(X=X, Z=Z, XX=XX, verb=0)
rmse4 <- sqrt(mean((fit4$ZZ.mean - ZZ)^2))
rmse4


###################################################
### code chunk number 15: tgp2.Rnw:345-346
###################################################
fit4$gpcs


###################################################
### code chunk number 16: tgp2.Rnw:357-360
###################################################
fit5 <-  btgpllm(X=X, Z=Z, XX=XX, basemax=10, verb=0)
rmse5 <- sqrt(mean((fit5$ZZ.mean - ZZ)^2))
rmse5 


###################################################
### code chunk number 17: cat-fb-mapt
###################################################
h <- fit1$post$height[which.max(fit1$posts$lpost)]
tgp.trees(fit5, "map")


###################################################
### code chunk number 18: tgp2.Rnw:377-378
###################################################
graphics.off()


###################################################
### code chunk number 19: tgp2.Rnw:409-412
###################################################
fit6 <-  btgpllm(X=X, Z=Z, XX=XX, basemax=10, splitmin=11, verb=0)
rmse6 <- sqrt(mean((fit6$ZZ.mean - ZZ)^2))
rmse6


###################################################
### code chunk number 20: tgp2.Rnw:448-449
###################################################
seed <- 0; set.seed(seed)


###################################################
### code chunk number 21: tgp2.Rnw:778-779
###################################################
f <- friedman.1.data(250) 


###################################################
### code chunk number 22: tgp2.Rnw:785-788
###################################################
Xf <- f[, 1:6] 
Zf <- f$Y 
sf <- sens(X=Xf, Z=Zf, nn.lhs=600, model=bgpllm, verb=0)


###################################################
### code chunk number 23: tgp2.Rnw:799-800
###################################################
names(sf$sens)


###################################################
### code chunk number 24: sens-full
###################################################
plot(sf, layout="sens", legendloc="topleft")


###################################################
### code chunk number 25: tgp2.Rnw:818-819
###################################################
graphics.off()


###################################################
### code chunk number 26: sens-mains
###################################################
par(mar=c(4,2,4,2), mfrow=c(2,3))
plot(sf, layout="sens", maineff=t(1:6))


###################################################
### code chunk number 27: tgp2.Rnw:846-847
###################################################
graphics.off()


###################################################
### code chunk number 28: sens-indices
###################################################
plot(sf, layout="sens", maineff=FALSE)


###################################################
### code chunk number 29: tgp2.Rnw:859-860
###################################################
graphics.off()


###################################################
### code chunk number 30: tgp2.Rnw:910-915
###################################################
X <- airquality[,2:4]
Z <- airquality$Ozone
rect <- t(apply(X, 2, range, na.rm=TRUE))
mode <- apply(X , 2, mean, na.rm=TRUE)
shape <- rep(2,3)


###################################################
### code chunk number 31: sens-udraw
###################################################
Udraw <- lhs(300, rect=rect, mode=mode, shape=shape)
par(mfrow=c(1,3), mar=c(4,2,4,2))
for(i in 1:3){
  hist(Udraw[,i], breaks=10,xlab=names(X)[i], 
       main="",ylab="", border=grey(.9), col=8) 
}  


###################################################
### code chunk number 32: tgp2.Rnw:928-929
###################################################
graphics.off()


###################################################
### code chunk number 33: tgp2.Rnw:941-943
###################################################
s.air <- suppressWarnings(sens(X=X, Z=Z, nn.lhs=300, rect=rect, 
                               shape=shape, mode=mode, verb=0))


###################################################
### code chunk number 34: sens-air1
###################################################
plot(s.air, layout="sens")


###################################################
### code chunk number 35: tgp2.Rnw:950-951
###################################################
graphics.off()


###################################################
### code chunk number 36: tgp2.Rnw:967-970
###################################################
rect[2,] <- c(0,5)
mode[2] <- 2
shape[2] <- 2


###################################################
### code chunk number 37: tgp2.Rnw:974-975
###################################################
sens.p <- suppressWarnings(sens(X=X,Z=Z,nn.lhs=300, model=NULL, rect=rect, shape=shape, mode=mode))


###################################################
### code chunk number 38: sens-air2
###################################################
s.air2 <- predict(s.air, BTE=c(1,1000,1), sens.p=sens.p, verb=0) 
plot(s.air2, layout="sens")


###################################################
### code chunk number 39: tgp2.Rnw:982-983
###################################################
graphics.off()


###################################################
### code chunk number 40: tgp2.Rnw:1006-1013
###################################################
X$Temp[X$Temp >70] <- 1
X$Temp[X$Temp >1] <- 0
rect <- t(apply(X, 2, range, na.rm=TRUE))
mode <- apply(X , 2, mean, na.rm=TRUE)
shape <- c(2,2,0)
s.air <- suppressWarnings(sens(X=X, Z=Z, nn.lhs=300, rect=rect, 
                               shape=shape, mode=mode, verb=0, basemax=2))


###################################################
### code chunk number 41: sens-air3
###################################################
plot(s.air, layout="sens")


###################################################
### code chunk number 42: tgp2.Rnw:1019-1020
###################################################
graphics.off()


###################################################
### code chunk number 43: tgp2.Rnw:1033-1034
###################################################
seed <- 0; set.seed(seed)


###################################################
### code chunk number 44: tgp2.Rnw:1207-1211
###################################################
rosenbrock <- function(x){ 
  x <- matrix(x, ncol=2)
  100*(x[,1]^2 - x[,2])^2 + (x[,1] - 1)^2 
}


###################################################
### code chunk number 45: tgp2.Rnw:1216-1217
###################################################
rosenbrock(c(1,1))


###################################################
### code chunk number 46: tgp2.Rnw:1225-1228
###################################################
rect <- cbind(c(-1,-1),c(5,5))
X <- lhs(40, rect)
Z <- rosenbrock(X)


###################################################
### code chunk number 47: tgp2.Rnw:1244-1246
###################################################
XX <- lhs(200, rect)
rfit <- bgp(X,Z,XX,improv=c(1,10), verb=0)


###################################################
### code chunk number 48: tgp2.Rnw:1254-1255
###################################################
cbind(rfit$improv,XX)[rfit$improv$rank <= 10,]


###################################################
### code chunk number 49: optim-fit1
###################################################
plot(rfit, as="improv")


###################################################
### code chunk number 50: tgp2.Rnw:1269-1270
###################################################
graphics.off()


###################################################
### code chunk number 51: optim-fit2
###################################################
rfit2 <- predict(rfit, XX=XX, BTE=c(1,1000,1), improv=c(5,20), verb=0) 
plot(rfit2, layout="as", as="improv")


###################################################
### code chunk number 52: tgp2.Rnw:1298-1299
###################################################
graphics.off()


###################################################
### code chunk number 53: tgp2.Rnw:1438-1439
###################################################
f <- function(x) { exp2d.Z(x)$Z }


###################################################
### code chunk number 54: tgp2.Rnw:1453-1456
###################################################
rect <- rbind(c(-2,6), c(-2,6))
X <- lhs(20, rect)
Z <- f(X)


###################################################
### code chunk number 55: tgp2.Rnw:1460-1473
###################################################
out <- progress <- NULL
for(i in 1:20) {
  
  ## get recommendations for the next point to sample
  out <- optim.step.tgp(f, X=X, Z=Z, rect=rect, prev=out, verb=0)

  ## add in the inputs, and newly sampled outputs
  X <- rbind(X, out$X)
  Z <- c(Z, f(out$X))
  
  ## keep track of progress and best optimum
  progress <- rbind(progress, out$progress)
}


###################################################
### code chunk number 56: optim-progress
###################################################
par(mfrow=c(1,2))
matplot(progress[,1:2], main="x progress",
        xlab="rounds", ylab="x[,1:2]", type="l", lwd=2)
legend("topright", c("x1", "x2"), lwd=2, col=1:2, lty=1:2)
plot(log(progress$improv), type="l", main="max log improv",
     xlab="rounds", ylab="max log(improv)")


###################################################
### code chunk number 57: tgp2.Rnw:1490-1491
###################################################
graphics.off()


###################################################
### code chunk number 58: tgp2.Rnw:1506-1507
###################################################
out$progress[1:2]


###################################################
### code chunk number 59: tgp2.Rnw:1532-1533
###################################################
formals(optim)$method


###################################################
### code chunk number 60: tgp2.Rnw:1537-1538
###################################################
formals(optim.ptgpf)$method


###################################################
### code chunk number 61: tgp2.Rnw:1552-1553
###################################################
seed <- 0; set.seed(seed)


###################################################
### code chunk number 62: tgp2.Rnw:1670-1673
###################################################
geo <- default.itemps(type="geometric")
har <- default.itemps(type="harmonic")
sig <- default.itemps(type="sigmoidal")


###################################################
### code chunk number 63: it-itemps
###################################################
par(mfrow=c(2,1))
all <- cbind(geo$k, har$k, sig$k)
matplot(all, pch=21:23,
        main="inv-temp ladders", xlab="indx", ylab="itemp")
legend("topright", pch=21:23, 
       c("geometric","harmonic","sigmoidal"), col=1:3)
matplot(log(all), pch=21:23,
        main="log(inv-temp) ladders", xlab="indx", ylab="itemp")


###################################################
### code chunk number 64: tgp2.Rnw:1688-1689
###################################################
graphics.off()


###################################################
### code chunk number 65: tgp2.Rnw:1755-1762
###################################################
ESS <- function(w)
{
  mw <- mean(w)
  cv2 <- sum((w-mw)^2)/((length(w)-1)*mw^2)
  ess <- length(w)/(1+cv2)
  return(ess)
}


###################################################
### code chunk number 66: tgp2.Rnw:1908-1911
###################################################
exp2d.data<-exp2d.rand() 
X<-exp2d.data$X 
Z<-exp2d.data$Z 


###################################################
### code chunk number 67: tgp2.Rnw:1917-1920
###################################################
its <- default.itemps(m=10)
exp.btlm <- btlm(X=X,Z=Z, bprior="b0", R=2, itemps=its, pred.n=FALSE,
                 BTE=c(1000,3000,2)) 


###################################################
### code chunk number 68: tgp2.Rnw:1945-1946
###################################################
exp.btlm$ess


###################################################
### code chunk number 69: tgp2.Rnw:1957-1960
###################################################
library(MASS)
moto.it <- btgpllm(X=mcycle[,1], Z=mcycle[,2], BTE=c(2000,52000,10),
        bprior="b0", R=3, itemps=geo, trace=TRUE, pred.n=FALSE, verb=0)


###################################################
### code chunk number 70: tgp2.Rnw:1964-1965
###################################################
moto.it$ess$combined


###################################################
### code chunk number 71: tgp2.Rnw:1969-1971
###################################################
p <- moto.it$trace$post
ESS(p$wlambda)


###################################################
### code chunk number 72: tgp2.Rnw:1977-1978
###################################################
ESS(p$w)


###################################################
### code chunk number 73: tgp2.Rnw:1983-1984
###################################################
as.numeric(c(sum(p$itemp == 1), moto.it$ess$each[1,2:3]))


###################################################
### code chunk number 74: tgp2.Rnw:1995-1997
###################################################
moto.reg <- btgpllm(X=mcycle[,1], Z=mcycle[,2], BTE=c(2000,52000,10),
        R=3, bprior="b0", trace=TRUE, pred.n=FALSE, verb=0)


###################################################
### code chunk number 75: tgp2.Rnw:2003-2006
###################################################
L <- length(p$height)
hw <- suppressWarnings(sample(p$height, L, prob=p$wlambda, replace=TRUE))
b <- hist2bar(cbind(moto.reg$trace$post$height, p$height, hw))


###################################################
### code chunk number 76: it-moto-height
###################################################
barplot(b, beside=TRUE, col=1:3, xlab="tree height", ylab="counts", 
         main="tree heights encountered")
legend("topright", c("reg MCMC", "All Temps", "IT"), fill=1:3)


###################################################
### code chunk number 77: tgp2.Rnw:2014-2015
###################################################
graphics.off()


###################################################
### code chunk number 78: it-moto-ktrace
###################################################
plot(log(moto.it$trace$post$itemp), type="l", ylab="log(k)", xlab="samples",
     main="trace of log(k)")


###################################################
### code chunk number 79: tgp2.Rnw:2048-2049
###################################################
graphics.off()


###################################################
### code chunk number 80: it-moto-khist
###################################################
b <- itemps.barplot(moto.it, plot.it=FALSE)
barplot(t(cbind(moto.it$itemps$counts, b)), col=1:2,
        beside=TRUE, ylab="counts", xlab="itemps", 
        main="inv-temp observation counts")
legend("topleft", c("observation counts", "posterior samples"), fill=1:2)


###################################################
### code chunk number 81: tgp2.Rnw:2080-2081
###################################################
graphics.off()


###################################################
### code chunk number 82: tgp2.Rnw:2104-2106
###################################################
moto.it.sig <- btgpllm(X=mcycle[,1], Z=mcycle[,2], BTE=c(2000,52000,10),
                      R=3, bprior="b0", krige=FALSE, itemps=sig, verb=0)


###################################################
### code chunk number 83: tgp2.Rnw:2110-2111
###################################################
moto.it.sig$ess$combined


###################################################
### code chunk number 84: it-moto-pred
###################################################
plot(moto.it.sig)


###################################################
### code chunk number 85: tgp2.Rnw:2117-2118
###################################################
graphics.off()


###################################################
### code chunk number 86: tgp2.Rnw:2144-2147
###################################################
Xcand <- lhs(10000, rbind(c(-6,6),c(-6,6)))
X <- dopt.gp(400, X=NULL, Xcand)$XX
Z <- exp2d.Z(X)$Z


###################################################
### code chunk number 87: tgp2.Rnw:2152-2154
###################################################
exp.reg <- btgpllm(X=X, Z=Z, BTE=c(2000,52000,10), bprior="b0", 
                   trace=TRUE, krige=FALSE, R=10, verb=0)


###################################################
### code chunk number 88: it-exp-pred
###################################################
plot(exp.reg)


###################################################
### code chunk number 89: tgp2.Rnw:2160-2161
###################################################
graphics.off()


###################################################
### code chunk number 90: tgp2.Rnw:2173-2175
###################################################
h <- exp.reg$post$height[which.max(exp.reg$posts$lpost)]
h


###################################################
### code chunk number 91: it-exp-mapt
###################################################
tgp.trees(exp.reg, "map")


###################################################
### code chunk number 92: tgp2.Rnw:2184-2185
###################################################
graphics.off()


###################################################
### code chunk number 93: tgp2.Rnw:2209-2212
###################################################
its <- default.itemps(k.min=0.02)
exp.it <- btgpllm(X=X, Z=Z, BTE=c(2000,52000,10), bprior="b0", 
               trace=TRUE, krige=FALSE, itemps=its, R=10, verb=0)


###################################################
### code chunk number 94: tgp2.Rnw:2217-2219
###################################################
exp.it$gpcs
exp.reg$gpcs


###################################################
### code chunk number 95: tgp2.Rnw:2227-2229
###################################################
p <- exp.it$trace$post
data.frame(ST=sum(p$itemp == 1), nIT=ESS(p$w), oIT=exp.it$ess$combined)


###################################################
### code chunk number 96: tgp2.Rnw:2241-2244
###################################################
L <- length(p$height)
hw <- suppressWarnings(sample(p$height, L, prob=p$wlambda, replace=TRUE))
b <- hist2bar(cbind(exp.reg$trace$post$height, p$height, hw))


###################################################
### code chunk number 97: it-exp-height
###################################################
barplot(b, beside=TRUE, col=1:3, xlab="tree height", ylab="counts", 
         main="tree heights encountered")
legend("topright", c("reg MCMC", "All Temps", "IT"), fill=1:3)


###################################################
### code chunk number 98: tgp2.Rnw:2252-2253
###################################################
graphics.off()


###################################################
### code chunk number 99: it-exp-trace-height
###################################################
ylim <- range(p$height, exp.reg$trace$post$height)
plot(p$height, type="l", main="trace of tree heights", 
     xlab="t", ylab="height", ylim=ylim)
lines(exp.reg$trace$post$height, col=2)
legend("topright", c("tempered", "reg MCMC"), lty=c(1,1), col=1:2)


###################################################
### code chunk number 100: tgp2.Rnw:2277-2278
###################################################
graphics.off()


###################################################
### code chunk number 101: it-expit-pred
###################################################
plot(exp.it)


###################################################
### code chunk number 102: it-expit-trees
###################################################
tgp.trees(exp.it, "map")


###################################################
### code chunk number 103: tgp2.Rnw:2305-2306
###################################################
graphics.off()


