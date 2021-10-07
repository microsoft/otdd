from setuptools import find_packages, setup

setup(
    name='otdd',
    version='0.1.0',
    description='Optimal Transport Dataset Distance',
    author='David Alvarez-Melis & Nicolo Fusi',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'numpy',
      'scipy',
      'matplotlib',
      'tqdm',
      'pot',
      'torch',
      'torchvision',
      'torchtext',
      'attrdict',
      'opentsne',
      'seaborn',
      'scikit-learn',
      'pandas',
      'geomloss',
      'munkres',
<<<<<<< HEAD
      'adjustText'
=======
      'adjustText',
>>>>>>> 60501b39492550cbe1de8e721c108c0672524ba7
    ],
    include_package_data=True,
    zip_safe=False
)
