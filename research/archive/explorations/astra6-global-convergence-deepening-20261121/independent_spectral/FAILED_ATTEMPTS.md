# Preserved failed attempts

1. The initial environment check `python -c "import sys,numpy; ..."` resolved to `C:\Users\liwei\AppData\Local\Microsoft\WindowsApps\python.exe` and failed before Python execution: the file could not be accessed by the system. No experiment ran and no numerical output was produced. The bundled Python interpreter was subsequently used; no package was installed.
2. The proposed extension of reverse-Bregman monotonicity to every SPD quadratic observation weight was rejected algebraically. The exact admissible 2-by-2 counterexample is preserved in `QUANTITATIVE_REVIEW.md` and checked in `results.json`. The failed implication was `C>=I, B>0 => <r,CB r>>=0`; the symmetric part of a product of two SPD matrices need not be PSD.
