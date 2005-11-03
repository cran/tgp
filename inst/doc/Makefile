all: tgp.pdf linear.pdf sin.pdf exp.pdf moto.pdf fried.pdf as.pdf clean

tgp.dvi: tgp.tex linear.tex sin.tex exp.tex moto.tex fried.tex as.tex
	latex tgp; bibtex tgp; latex tgp; latex tgp; 

tgp.ps: tgp.dvi
	dvips -o -t letter tgp

linear.ps: tgp.dvi
	dvips -pp 14-15 -t letter tgp -o linear.ps

sin.ps: tgp.dvi
	dvips -pp 15-19 -t letter tgp -o sin.ps

exp.ps: tgp.dvi
	dvips -pp 19-21 -t letter tgp -o exp.ps

moto.ps: tgp.dvi
	dvips -pp 21-25 -t letter tgp -o moto.ps

fried.ps: tgp.dvi
	dvips -pp 24-26 -t letter tgp -o fried.ps

as.ps: tgp.dvi
	dvips -pp 26-28 -t letter tgp -o as.ps

tgp.pdf: tgp.ps
	ps2pdf tgp.ps

linear.pdf: linear.ps
	ps2pdf linear.ps

sin.pdf: sin.ps
	ps2pdf sin.ps

exp.pdf: exp.ps
	ps2pdf exp.ps

moto.pdf: moto.ps
	ps2pdf moto.ps

fried.pdf: fried.ps
	ps2pdf fried.ps

as.pdf: as.ps
	ps2pdf as.ps

clean: 
	rm -f *.log *.aux *.eps *.bak *.rel *.bbl *.blg \
		*.dvi *.ps *.tex *-*.pdf
