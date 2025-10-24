import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calapy",
    version="0.2.0.3",
    author="Carmelo Calafiore",
    author_email="dr.carmelo.calafiore@gmail.com",
    description="Personal Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/calapy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "opencv-contrib-python",
        "matplotlib",
        "selenium",
        "pygame",
        "torch",
        "torchvision",
        "Pillow",
    ],
    python_requires='>=3.6')

# examples of the kw "install_requires"
# setup(
#     #...,
#     install_requires = [
#         'docutils',
#         'BazSpam ==1.1',
#         "enum34;python_version<'3.4'",
#         "pywin32 >= 1.0;platform_system=='Windows'"]
#     #...)

# more info the kw "install_requires" at the link below
# at https://setuptools.readthedocs.io/en/latest/userguide/dependency_management.html#declaring-dependencies
