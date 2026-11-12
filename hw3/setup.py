from setuptools import find_packages, setup

package_name = 'hw3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/hw3.launch.py',
            'launch/dead_reckoning.launch.py',
            'launch/motor_mapping.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chenghao Li',
    maintainer_email='chl235@ucsd.edu',
    description='AprilTag localization and robot hardware nodes',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_control = hw3.motor_control:main',
            'velocity_mapping = hw3.velocity_mapping:main',
            'ekf_waypoint_controller = '
            'hw3.ekf_waypoint_controller:main',
            'dead_reckoning_waypoint_controller = '
            'hw3.dead_reckoning_waypoint_controller:main',
            'ekf_localizer = hw3.ekf_localizer:main',
            'tag_position_logger = hw3.tag_position_logger:main',
            'camera_position_inspector = '
            'hw3.camera_position_inspector:main',
            'calibrate_angular_motion = '
            'hw3.calibrate_angular_motion:main',
        ],
    },
)
