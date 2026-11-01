# Original MATLAB program

`matlab/` preserves the files from the authors' `海阔天空程序.ZIP` archive.
The workflow is `Mapping_plus.m` → `Stop_Selection.m` → `Route_Finding.m`, with data passed
through `temp.mat` and `temp2.mat`. The original instruction document is stored as
`original-instructions.docx`.

[manifest.json](manifest.json) records the archive hash and each extracted file's SHA-256.
Original file contents and notices are preserved. The repository's MIT grant covers the
Python implementation and accompanying project materials; these original MATLAB files
retain their existing licensing notices.

`temp.mat` contains the 15-target example; `temp2.mat` contains its selected seven-target
anchor group. Input units and portable dataset descriptions are in
[examples/README.md](../examples/README.md).
