
"plot.tgp" <-
function(x, pparts=TRUE, proj=NULL, slice=NULL, map=NULL, as=NULL,
         layout="both", main=NULL, xlab=NULL, ylab=NULL, zlab=NULL,
         pc="pc", method="loess", gridlen=40, span=0.1, ...)
{ 


   if( x$d==2 ){

    X <- rbind(x$X,x$XX)
    o<-order(X[,2])
    X<- X[o,]
    f<-X[,1]==1
    c<-X[,1]==0

    Zp <- c(x$Zp.mean, x$ZZ.mean)[o]
    Zp.q1 <- c(x$Zp.q1, x$ZZ.q1)[o]
    Zp.q2 <- c(x$Zp.q2, x$ZZ.q2)[o]
     
    plot(x$X[x$X[,1]==0,2],x$Z[x$X[,1]==0], ylim=range(c(Zp,x$Z)),
         xlab="", ylab="",
         main=main)
    lines(x$X[x$X[,1]==1,2],x$Z[x$X[,1]==1], type="p", pch=2,col="blue")

    lines(X[c,2], Zp[c])


    lines(X[f,2], Zp[f], col=2)
    lines(X[f,2], Zp.q1[f], col=4, lty=3)
    lines(X[f,2], Zp.q2[f], col=4, lty=3)
    lines(X[c,2], Zp.q1[c], col=5, lty=3)
    lines(X[c,2], Zp.q2[c], col=5, lty=3)
    if(pparts) tgp.plot.parts.1d(x$parts[,2])
   
   }
   else{
    par( mfrow=c(1,2) )
    if(is.null(proj)) proj <- c(1,2)
    proj <- proj+1
    X <- rbind(as.matrix(x$X), x$XX)
    Z.mean <- c(x$Zp.mean, x$ZZ.mean)
   
    c<-X[,1]==0
    f<-X[,1]==1
    Xc <- X[c,proj]
    Xf <- X[f,proj]
    Zc.mean <- Z.mean[c]
    Zf.mean <- Z.mean[f]

    nXc <- dim(Xc)[1]
    pc <- seq(1,nXc)
    nXf <- dim(Xf)[1]
    pf <- seq(1,nXf)
    dX <- dim(X)[2]

    slice.image(Xc[,1],Xc[,2],z=Zc.mean,xlab="",ylab="",main="Coarse",
                  method=method,gridlen=gridlen,span=span,...)
    points(x$X[c,proj], ...)
    points(x$X[f,proj],pch=2,col=2, ...)

    slice.image(Xf[,1],Xf[,2],z=Zf.mean,xlab="",ylab="",main="Fine",
                  method=method,gridlen=gridlen,span=span,...)
    points(x$X[c,proj], ...)
    points(x$X[f,proj],pch=2,col=2, ...)
   }
   
}
