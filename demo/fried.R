###################################################
### chunk number 1: 
###################################################
library(tgp)
graphics.off()


###################################################
### chunk number 2: 
###################################################
f <- friedman.1.data(200)
ff <- friedman.1.data(1000)
X <- f[,1:10]; Z <- f$Y
XX <- ff[,1:10]


###################################################
### chunk number 3: 
###################################################
fr.btlm <- btlm(X=X, Z=Z, XX=XX, tree=c(0.95,2,10), m0r1=TRUE)
fr.btlm.mse <- sqrt(mean((fr.btlm$ZZ.mean - ff$Ytrue)^2))
fr.btlm.mse


###################################################
### chunk number 4: 
###################################################
fr.bgpllm <- bgpllm(X=X, Z=Z, XX=XX, m0r1=TRUE)
fr.bgpllm.mse <- sqrt(mean((fr.bgpllm$ZZ.mean - ff$Ytrue)^2))
fr.bgpllm.mse


