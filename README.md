## Internal Library for the ITIV Lab LAMA

Link to the Lab: [Lab on ITIV Website](https://www.itiv.kit.edu/60_LAMA.php)

This library mainly delivers functions for checkpoint creating and verification so that the students can check whether their function returns the expected result.
Checkpoints can be created by the sample solution and can be pushed onto an online server that can be accessed at `https://hdd1.itiv.kit.edu/tools/lama`.
If checkpoints are not available online, local checkpoints in a `checkpoint` folder are a fallback.

In addition, the library offers the delete wrapper `lama_del_wrap` that allows to have incomplete statements in the student workbook.
For example, `test = lama_del_wrap(123)` becomes `test = ...` after the student version generator ran.
Note, that this function does not modify any data and does not work with tuples yet.


### Installation Guide

Install as editable: `pip install -e dir`, where dir represents the directory that this folder was cloned to.

Normal install: `pip install .` in this directory, when the library is cloned or `pip install 'lama @ git+https://github.com/itiv-kit/lama-python-lib@COMMIT ID'`

