"check.matrix" <- 
function(X, Z=NULL)
{
	# format X
	if(is.null(X)) return(NULL)
	n <- dim(X)[1]
	if(is.null(n)) { n <- length(X); X <- matrix(X, nrow=n) }
	X <- as.matrix(X)

	# if a Z is provided to go along with X
	if(!is.null(Z)) {

		# format Z
		Z <- as.vector(matrix(Z, ncol=1)[,1])
		if(length(Z) != n) stop("mismatched row dimension in X and Z")

		# calculate locations of NAs NaNs and Infs in Z
		nna <- (1:n)[!is.na(Z) == 1]
		nnan <- (1:n)[!is.nan(Z) == 1]
		ninf <- (1:n)[!is.infinite(Z) == 1]
		if(length(nna) < n) warning(paste(n-length(nna), "NAs removed from input vector"))
		if(length(nnan) < n) warning(paste(n-length(nnan), "NaNs removed from input vector"))
		if(length(ninf) < n) warning(paste(n-length(ninf), "Infs removed from input vector"))

		neitherZ <- intersect(nna, intersect(nnan, ninf))
	} else neitherZ <- (1:n)

	# calculate row locations of NAs NaNs and Infs in X
	nna <- (1:n)[apply(!is.na(X), 1, prod) == 1]
	nnan <- (1:n)[apply(!is.nan(X), 1, prod) == 1]
	ninf <- (1:n)[apply(!is.infinite(X), 1, prod) == 1]
	if(length(nna) < n) warning(paste(n-length(nna), "NAs removed from input matrix"))
	if(length(nnan) < n) warning(paste(n-length(nnan), "NaNs removed from input matrix"))
	if(length(ninf) < n) warning(paste(n-length(ninf), "Infs removed from input matrix"))
	neitherX <- intersect(nna, intersect(nnan, ninf))

	# combine good X and Z rows
	neither <- intersect(neitherZ, neitherX)
	X <- matrix(X[neither,], nrow=length(neither))
	Z <- Z[neither]

	return(list(X=X, Z=Z))
}
