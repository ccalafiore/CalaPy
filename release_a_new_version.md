# Release Check List

1. Delete 2 folders: [calapy.egg-info](calapy.egg-info), [dist](dist).

2. Choose the name of the new version with the format A.B.C.D where A, B, C and D are positive integers. Increase D by 1
   if the new version only has bag fixes and marginal changes. Increase C by 1 and set D to 0 if the new version has
   some new features, but any code writen with previous version still work. Increase B by 1 and set C and D to 0, if
   some code that was writen with the previous version may not work, but the main structure of the package is the same.
   Increase A by 1 and set B, C and D to 0, if the main structure of the package has changed and most of the code that
   was writen with the previous code does not work any longer.

3. Update the version and release date in [calapy/__init__.py](src/calapy/__init__.py).

4. Update the version in [pyproject.toml](pyproject.toml).

5. Activate conda environment with:
   
   1. Install anaconda
   2. Create a conda environment with:
      ```
      conda create --name myenv
      ```
   3. Activate the environment with:
      ```
      conda activate myenv
      ```

6. Open Command Prompt as administrator and run the following commands:

   1. Update these modules:
      ```
      python -m pip install --upgrade pip build twine
      ```
   2. Change working directory to the repo's with:
      ```
      cd "directory\of\CalaPy"
      ```

   3. Run these:
      ```
      python -m build
      python -m twine upload --repository pypi dist/*
      ```
   4. Close Command Prompt

7. Uninstall calapy from an environment myenv:
   ```
   conda activate myenv
   python -m pip uninstall calapy
   ```
   If they still exist, manually delete the 2 directories:
   - directory\of\anaconda\envs\myenv\Lib\site-packages\calapy
   - directory\of\anaconda\envs\myenv\Lib\site-packages\calapy-\*.\*.\*.\*.dist-info

7. Re-install calapy with:
   ```
   python -m pip install --upgrade calapy
   ```

