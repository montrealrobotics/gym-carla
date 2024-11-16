from setuptools import setup, find_packages

setup(name='gym_carla',
      version='0.0.1',
      install_requires=['gym', 'pygame'],
      package_dir={"": "gym_carla"},
      packages=find_packages("gym_carla"),
)
