from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''Return all the requirements from the requirements file.'''
    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements]
            if "-e ." in requirements:
                requirements.remove("-e .")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        exit(1)
    
    return requirements

setup(
    name="student performance project",
    author="Karan",
    author_email="shinigami709k@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
