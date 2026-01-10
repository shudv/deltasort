.PHONY: all clean rebuild

all: article.pdf

article.pdf: article/main.tex content.tex refs.bib figures/**/* sections/*.tex
	pdflatex -interaction=nonstopmode article/main.tex
	bibtex article/main
	pdflatex -interaction=nonstopmode article/main.tex
	pdflatex -interaction=nonstopmode article/main.tex
rebuild: clean all

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz main.pdf
