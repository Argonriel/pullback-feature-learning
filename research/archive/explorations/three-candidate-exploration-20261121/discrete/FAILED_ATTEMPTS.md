# Failed attempt record

The initial source snapshot is `failed_initial_source.py`. The first execution was

`C:\Users\liwei\Documents\ChatGPT\277\.venv\Scripts\python.exe experiments\three_candidate_exploration_20261121\discrete\factor_step_cancellation.py --output ..\tmp\three-candidate-exploration-20261121\discrete\run01`

It exited before creating the output directory or running any calculation:

```
Traceback (most recent call last):
  File "C:\Users\liwei\Documents\ChatGPT\277\pullback-feature-learning\experiments\three_candidate_exploration_20261121\discrete\factor_step_cancellation.py", line 12, in <module>
    import mpmath as mp
ModuleNotFoundError: No module named 'mpmath'
```

The implementation was changed to the standard-library 80-digit `decimal` type and a direct Newton solve for the scalar continuous endpoint. No package was installed and no experimental parameter or stopping rule was changed.
