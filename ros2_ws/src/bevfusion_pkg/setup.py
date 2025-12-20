from setuptools import find_packages, setup

package_name = 'bevfusion_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_node = bevfusion_pkg.bev_node:main',
            'replay_node = bevfusion_pkg.replay_node:main',
            'detection_logger_node = bevfusion_pkg.detection_logger_node:main',
        ],
    },
)
