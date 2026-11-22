# Appendix B. Reproducing every number

```bash
git clone https://github.com/Daniellopesba/shadow-feature-selection
cd shadow-feature-selection
python -m venv .venv && . .venv/bin/activate
pip install -e . -r requirements-experiments.txt   # pinned versions

pytest                                  # unit tests, incl. numerical checks of Props 1, 2, 4-6, 8, 9

cd experiments
python e1_first_split.py                # about 15 min on 4 cores
python e2_sweep.py                      # about 15 min
python e2_sweep.py --n-noise 20         # about 3 min
python e3_boosting_rounds.py            # about 1 min
python e4_mitigations.py                # about 45 min
python e5_backends.py                   # about 15 min
python e6_univariate.py                 # about 2 min
python e7_rounds.py                     # about 30 min
python e8_many_shadows.py               # about 2 min
python make_figures.py                  # figures/ and figures/tables.md
python paper_numbers.py                 # results/paper_numbers.md (Appendix A numbers)

cd ../paper && ./build.sh               # paper/build/paper.docx and paper.html
```

Every script accepts `--quick` for a tiny profile, which writes `*_quick`
files and never overwrites the paper's results. Per-repeat seeds are an MD5
hash of the scenario parameters and the repeat index
(`shadowselect.simulation.derive_seed`). Each `results/*.meta.json` records
the library versions and wall time of its run.
