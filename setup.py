from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> list[str]:
    """
    This function reads the requirements from a file and returns them as a list.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]

        # If '-e .' is present, remove it
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="MLProject",
    version="0.0.1",
    author="Tarun",
    author_email="tk9312265953@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)