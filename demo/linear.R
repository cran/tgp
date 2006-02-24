###################################################
### chunk number 1: 
###################################################
library(tgp)
library(maptree)
graphics.off()


###################################################
### chunk number 2: 
###################################################
# 1-d linear data input and predictive data
X <- seq(0,1,length=50)  # inputs
XX <- seq(0,1,length=99) # predictive locations
Z <- 1 + 2*X + rnorm(length(X),sd=0.25) # responses


###################################################
### chunk number 3: 
###################################################
lin.blm <- blm(X=X, XX=XX, Z=Z)


###################################################
### chunk number 4: blm
###################################################
plot(lin.blm, main='Linear Model,', layout='surf')
abline(1,2,lty=3,col='blue')


###################################################
### chunk number 5: 
###################################################
rl <- readline("press RETURN to continue: ")
dev.off()


###################################################
### chunk number 6: 
###################################################
lin.gpllm <- bgpllm(X=X, XX=XX, Z=Z)


###################################################
### chunk number 7: gplm
###################################################
plot(lin.gpllm, main='GP LLM,', layout='surf')
abline(1,2,lty=4,col='blue')


