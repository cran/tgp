all: all.pdf clean

all.pdf: tgp.pdf tgp2.pdf

TEX	= tgp.tex tgp2.tex
PS 	= tgp.ps tgp2.ps

tgp.dvi: tgp.tex
	latex tgp; bibtex tgp; latex tgp; latex tgp; 

tgp2.dvi: tgp2.tex
	latex tgp2; bibtex tgp2; latex tgp2; latex tgp2; 

tgp.ps: tgp.dvi
	dvips -Pwww -o -t letter tgp

tgp2.ps: tgp2.dvi
	dvips -Pwww -o -t letter tgp2

tgp.pdf: tgp.ps
	ps2pdf tgp.ps

tgp2.pdf: tgp2.ps
	ps2pdf tgp2.ps

clean: 
	rm -f *.log *.aux *-*.eps *.bak *.rel *.bbl *.blg \
		*.dvi ${TEX} *-*.pdf ${PS}
