# Supplied MATLAB source

`matlab/` is extracted from the author-supplied `海阔天空程序.ZIP`, a small modification of
the paper's program. Original file bytes are preserved. The Chinese instruction document
has been renamed `original-instructions.docx`; its text identifies `Mapping_plus` as the
first stage, followed by `Stop_Selection`.

Typical workflow: `Mapping_plus.m` → `Stop_Selection.m` → `Route_Finding.m`.
The scripts communicate using `temp.mat` and `temp2.mat`, require MATLAB plotting and image
processing functions, and plan the largest anchor group in the supplied modified workflow.

Some K-means helper files contain educational exercise comments. Their notices are preserved;
the repository's new MIT grant does not relicense these legacy files. The independently
implemented Python package does not import or execute them.

`temp.mat` contains 15 targets with workload and integer sample demand. `temp2.mat` contains
one selected seven-target anchor group. Neither is the paper's original 100-target simulation.
