all: all.pdf clean

all.pdf: tgp.pdf

PS: 	tgp.ps

TEX	= tgp.tex
tgp.dvi:  ${TEX}
	latex tgp; bibtex tgp; latex tgp; latex tgp; 

tgp.ps: tgp.dvi
	dvips -Pwww -o -t letter tgp

tgp.pdf: tgp.ps
	ps2pdf tgp.ps

clean: 
	rm -f *.log *.aux *-*.eps *.bak *.rel *.bbl *.blg \
		*.dvi ${TEX} *-*.pdf ${PS}
