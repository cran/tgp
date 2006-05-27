"exp2d.rand" <-
function(n1=50, n2=30)
{
	if(n1 < 0 || n2 < 0) stop("n1 and n1 must be >= 0")
	if(n1 + n2 >= 441) stop("n1 + n2 must bbe <= 441")

	data(exp2d); n <- dim(exp2d)[1]
	si <- (1:n)[1==apply(exp2d[,1:2] <= 2, 1, prod)]
	s <- c(sample(si, size=n1, replace=FALSE), 
	sample(setdiff(1:n, si), n2, replace=FALSE))
	X <- as.matrix(exp2d[s,1:2]);
	ss <- setdiff(1:n, s)
	XX <- exp2d[ss, 1:2];
	Z <- as.vector(exp2d[s,3]);
	Ztrue <- as.vector(exp2d[s,4]);
	ZZ <- as.vector(exp2d[ss,3]);
	ZZtrue <- as.vector(exp2d[ss,4]);

	return(list(X=X, Z=Z, Ztrue=Ztrue, XX=XX, ZZ=ZZ, ZZtrue=ZZtrue))
}
