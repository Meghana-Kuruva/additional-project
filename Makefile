
all: pdf 

clean:
	rm -f *~
	rm -f *.dvi *.aux *.log *.bbl *.blg *.bak *.brf *.lbl *.out

pdf: ICVGIP-Latex-Template.tex
	pdflatex ICVGIP-Latex-Template.tex
	bibtex ICVGIP-Latex-Template
	pdflatex ICVGIP-Latex-Template.tex
	pdflatex ICVGIP-Latex-Template.tex
	pdflatex ICVGIP-Latex-Template.tex

