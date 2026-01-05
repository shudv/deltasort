.PHONY: all clean rebuild

all: main.pdf sea/main.pdf

main.pdf: main.tex refs.bib figures/**/*
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

sea/main.pdf: sea/main.tex refs.bib figures/**/*
	cd sea && pdflatex -interaction=nonstopmode main.tex
	cd sea && bibtex main
	cd sea && pdflatex -interaction=nonstopmode main.tex
	cd sea && pdflatex -interaction=nonstopmode main.tex

rebuild: clean all

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz main.pdf
	rm -f sea/*.aux sea/*.bbl sea/*.blg sea/*.log sea/*.out sea/*.toc sea/*.fls sea/*.fdb_latexmk sea/*.synctex.gz sea/main.pdf
