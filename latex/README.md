# LaTeX Research Paper

This folder contains the complete LaTeX source for the MelMOT research paper.

## Contents

- **main.tex** - Main LaTeX document with the complete research paper
- **references.bib** - Bibliography file with all citations
- **titlepage.sty** - Custom LaTeX style for the title page
- **pictures/** - All figures, diagrams, and images used in the paper

## Compilation

To compile the research paper:

```bash
# Navigate to this directory
cd latex

# Compile with bibliography
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Paper Information

- **Title**: Innovative Approaches to Multi-User Tracking in Retail Spaces
- **Author**: Ivan Novosad
- **Topic**: Multi-Object Tracking and Cross-Camera Re-identification
- **Focus**: Single-camera MOT and cross-camera Re-ID methodologies

## Figures

The `pictures/` folder contains all visual elements including:
- Architecture diagrams
- Performance comparisons
- Visualization results
- Model schematics
- Experimental results

## Note

This is the research paper source code. The compiled PDF is available at:
[Download Research Paper](https://drive.google.com/file/d/1r1hOHQpZdUl5fumM93CqdrHqX0_uswO0/view)
