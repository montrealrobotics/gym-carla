from setuptools import setup, find_packages

setup(name='gym_carla',
      version='0.0.1',
      install_requires=['gym', 'pygame'],
      package_dir={"": "src"},
      packages=find_packages("src"),
)
