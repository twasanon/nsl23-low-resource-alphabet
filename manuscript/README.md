# Manuscript

Source: `nsl23_low_resource.tex`  
Compiled PDF: `nsl23_low_resource.pdf`  
Tables: `generated_tables.tex` (frozen 30-run matrix) and `generated_improvement_tables.tex`  
Figures: `figures/` (copied from `artifacts/figures/`)

```bash
python ../scripts/write_latex_tables.py
python ../scripts/generate_improvement_figures.py
cp ../artifacts/figures/figure_{1,2,3,4,5,6,7,9,10,11,12}_*.pdf figures/ 2>/dev/null || true
tectonic -X compile nsl23_low_resource.tex
```
