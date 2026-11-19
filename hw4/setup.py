from glob import glob
from setuptools import find_packages, setup

package_name = 'hw4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'LICENSE']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chenghao Li',
    maintainer_email='chl235@ucsd.edu',
    description='HW4 path planning',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
