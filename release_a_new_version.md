# Release Check List

1. Delete 4 folders: "build"; "calapy.egg-info"; "dist".

2. Update the version and release date in calapy/__init__.py.

3. Update the version in setup.py.

4. Activate conda environment with:
   
   1. Install anaconda
   2. Create a conda environment with:
      ```
      conda create --name calapy
      ```
   3. Activate the environment with:
      ```
      conda activate calapy
      ```

5. Open Command Prompt as administrator and run the following commands:

   1. Update these modules:
      ```
      python -m pip install --upgrade pip
      python -m pip install --upgrade setuptools wheel
      python -m pip install --upgrade twine
      ```
   2. Change working directory to the repo's with:
      ```
      cd "directory\of\CalaPy"
      ```

   3. Run these:
      ```
      python setup.py sdist bdist_wheel
      python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
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

7. Install calapy with:
   ```
   python -m pip install --upgrade calapy
   ```

