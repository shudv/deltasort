PAPER = main

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex sections/*.tex refs.bib figures/headline.pdf
	pdflatex -interaction=nonstopmode $(PAPER)
	bibtex $(PAPER) || true
	pdflatex -interaction=nonstopmode $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)

figures/headline.pdf: scripts/make_figures.py ../rust/results/validated.csv
	cd .. && python3 paper2/scripts/make_figures.py

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out sections/*.aux

distclean: clean
	rm -f $(PAPER).pdf figures/*.pdf figures/*.png

.PHONY: all clean distclean
