###################################################
### chunk number 1: 
###################################################
library(tgp)
library(maptree)
graphics.off()


###################################################
### chunk number 2: 
###################################################
exp2d.data <- exp2d.rand()
X <- exp2d.data$X; Z <- exp2d.data$Z
Xcand <- exp2d.data$XX


###################################################
### chunk number 3: 
###################################################
exp1 <- btgpllm(X=X, Z=Z, pred.n=FALSE, corr="exp")


###################################################
### chunk number 4: mapt
###################################################
tgp.trees(exp1)


###################################################
### chunk number 5: 
###################################################
rl <- readline("press RETURN to continue: ")
dev.off()


###################################################
### chunk number 6: 
###################################################
XX <- tgp.design(10, Xcand, exp1)


###################################################
### chunk number 7: cands
###################################################
plot(exp1$X, pch=19, cex=0.5); points(XX)
tgp.plot.parts.2d(exp1$parts)


###################################################
### chunk number 8: 
###################################################
rl <- readline("press RETURN to continue: ")
dev.off()


###################################################
### chunk number 9: 
###################################################
exp1.btgpllm <- btgpllm(X=X, Z=Z, XX=XX, corr="exp", ego=TRUE, ds2x=TRUE)


###################################################
### chunk number 10: expalm
###################################################
par(mfrow=c(1,2), bty="n")
plot(exp1.btgpllm, main="treed GP LLM,", method="akima", layout="surf")
plot(exp1.btgpllm, main="treed GP LLM,", method="akima", layout="as", as="alm")


###################################################
### chunk number 11: 
###################################################
rl <- readline("press RETURN to continue: ")
dev.off()


###################################################
### chunk number 12: expalcego
###################################################
par(mfrow=c(1,2), bty="n")
plot(exp1.btgpllm, main="treed GP LLM,", method="akima", layout='as', as='alc')
plot(exp1.btgpllm, main="treed GP LLM,", method="akima", layout='as', as='ego')

