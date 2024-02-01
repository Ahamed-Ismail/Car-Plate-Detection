from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path,'r') as f_obj:
        requirements=f_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        return requirements

setup(
    name="CarPlateDetection",
    version='0.0.1',
    author="ahamed ismail hisam m",
    install_requires=get_requirements('requirements.txt')
    packages=find_packages()
)