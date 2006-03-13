print.tgp <- function(x, ...)
{
	cat("This is a 'tgp' class object\n")
	cat("It is basically a list with the following entries:\n\n")
	print(names(x), quote=FALSE)

	cat("\nSee tgp or btgp for an explanation of the individual entries\n")
	cat("See plot.tgp and tgp.trees for help with visualization\n")
}
