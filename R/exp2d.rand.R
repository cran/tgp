"exp2d.rand" <-
function()
{
	data(exp2d); n <- dim(exp2d)[1]
	si <- (1:n)[1==apply(exp2d[,1:2] <= 2, 1, prod)]
	s <- c(sample(si, size=50, replace=FALSE), 
	sample(setdiff(1:n, si), 30, replace=FALSE))
	X <- as.matrix(exp2d[s,1:2]);
	ss <- setdiff(1:n, s);
	XX <- exp2d[ss, 1:2];
	Z <- as.vector(exp2d[s,3]);

	return(list(X=X, Z=Z, XX=XX))
}
