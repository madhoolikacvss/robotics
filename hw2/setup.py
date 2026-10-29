from setuptools import find_packages, setup

package_name = 'hw2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hw2.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chenghao Li',
    maintainer_email='chl235@ucsd.edu',
    description='AprilTag waypoint controller and robot hardware nodes',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_control = hw2.motor_control:main',
            'velocity_mapping = hw2.velocity_mapping:main',
            'apriltag_waypoint_controller = hw2.apriltag_waypoint_controller:main',
        ],
    },
)
