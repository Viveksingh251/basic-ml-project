from setuptools import find_packages, setup 
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirement=[]
    with open(file_path) as file_obj:
        requirement=file_obj.readlines()
        requirement=[req.replace("\n","") for req in requirement]
        if '-e .' in requirement:
            requirement.remove('-e .')

setup(
name="ML project",
version='0.0.1',
author='Vivek singh',
author_email='viveks04251@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirement.txt')





)