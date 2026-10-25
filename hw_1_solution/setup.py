from setuptools import find_packages, setup

package_name = 'hw_1_solution'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hw1_solution.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chenghao Li',
    maintainer_email='chl235@ucsd.edu',
    description='Homework 1 solution package',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_control = hw_1_solution.motor_control:main',
            'velocity_mapping = hw_1_solution.velocity_mapping:main',
            'hw1_solution = hw_1_solution.hw1_solution:main',
        ],
    },
)
