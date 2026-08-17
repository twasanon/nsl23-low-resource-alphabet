# Manuscript

Source: `nsl23_low_resource.tex`  
Compiled PDF: `nsl23_low_resource.pdf`  
Tables: `generated_tables.tex` (from `scripts/write_latex_tables.py`)  
Figures: `figures/` (copied from `artifacts/figures/`)

```bash
python ../scripts/write_latex_tables.py
cp ../artifacts/figures/figure_{1,2,3,4,5}_*.pdf figures/ 2>/dev/null || true
tectonic -X compile nsl23_low_resource.tex
```
