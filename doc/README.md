# ReNomIMG Document

### Requirements
- sphinx
- sphinxcontrib-versioning==2.2.1

### Write document
- Write documents of python module to the source.
- Put Japanese translation \_locale/ja/.

### Extract docstrings from source code using Autodoc.

Following command updates rst file. **Be careful for using these commands since 
they will override rst files. You should specify the path for create doc.**
If you have not added any new method, you don't need to run following command.

```sphinx-apidoc -o . ../renom_img/api```

For initial extraction, use ```-F``` option for full reconstructing docments.

```sphinx-apidoc -F -o . ../renom_img/api```

### Translation files.
For creating translation template files, use following command.

### Build document on current branch

### Build document multiple branch with sphinx-versioning
