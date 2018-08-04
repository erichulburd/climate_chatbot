from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensorlayer==1.9',
  'tensorflow==1.9',
  'google-cloud-storage',
  'nltk',
  'numpy'
]

setup(
    name='climate_chatbot',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)
