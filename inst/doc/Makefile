all: tgp.pdf linear.pdf sin.pdf exp.pdf moto.pdf fried.pdf as.pdf traces.pdf pred.pdf clean

all.pdf: tgp.pdf linear.pdf sin.pdf exp.pdf moto.pdf fried.pdf as.pdf traces.pdf pred.pdf

tgp.dvi: tgp.tex linear.tex sin.tex exp.tex moto.tex fried.tex as.tex traces.tex pred.tex
	latex tgp; bibtex tgp; latex tgp; latex tgp; 

tgp.ps: tgp.dvi
	dvips -Pwww -o -t letter tgp

linear.ps: tgp.dvi
	dvips -Pwww -pp 14-17 -t letter tgp -o linear.ps

sin.ps: tgp.dvi
	dvips -Pwww -pp 17-20 -t letter tgp -o sin.ps

exp.ps: tgp.dvi
	dvips -Pwww -pp 20-25 -t letter tgp -o exp.ps

moto.ps: tgp.dvi
	dvips -Pwww -pp 23-28 -t letter tgp -o moto.ps

fried.ps: tgp.dvi
	dvips -Pwww -pp 28-29 -t letter tgp -o fried.ps

as.ps: tgp.dvi
	dvips -Pwww -pp 30-33 -t letter tgp -o as.ps

traces.ps: tgp.dvi
	dvips -Pwww -pp 33-38 -t letter tgp -o traces.ps

pred.ps: tgp.dvi
	dvips -Pwww -pp 39-41 -t letter tgp -o pred.ps

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

traces.pdf: traces.ps
	ps2pdf traces.ps

pred.pdf: pred.ps
	ps2pdf pred.ps

clean: 
	rm -f *.log *.aux *.eps *.bak *.rel *.bbl *.blg \
		*.dvi *.ps *.tex *-*.pdf
