.PHONY: all clean rebuild

all: main.pdf

main.pdf: main.tex refs.bib figures/**/* sections/*.tex
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

rebuild: clean all

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz main.pdf
