# Release Check List

1. Delete 2 folders: [calapy.egg-info](calapy.egg-info), [dist](dist).

2. Update the version and release date in [calapy/__init__.py](src/calapy/__init__.py).

3. Update the version in [pyproject.toml](pyproject.toml).

4. Activate conda environment with:
   
   1. Install anaconda
   2. Create a conda environment with:
      ```
      conda create --name myenv
      ```
   3. Activate the environment with:
      ```
      conda activate myenv
      ```

5. Open Command Prompt as administrator and run the following commands:

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

6. Uninstall calapy from an environment myenv:
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

