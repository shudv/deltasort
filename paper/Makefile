.PHONY: all clean rebuild

all: article.pdf sea.pdf

article.pdf: article.tex refs.bib figures/**/*
	pdflatex -interaction=nonstopmode article.tex
	bibtex article
	pdflatex -interaction=nonstopmode article.tex
	pdflatex -interaction=nonstopmode article.tex

sea.pdf: sea.tex refs.bib figures/**/* lipics-v2021.cls
	pdflatex -interaction=nonstopmode sea.tex
	bibtex sea
	pdflatex -interaction=nonstopmode sea.tex
	pdflatex -interaction=nonstopmode sea.tex

rebuild: clean all

clean:
	rm -rf *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.vtc article.pdf sea.pdf out/*