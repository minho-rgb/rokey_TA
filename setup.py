from setuptools import find_packages, setup

package_name = 'tracking'

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
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test = tracking.test:main',
            'track = tracking.tracking:main',
            'track_f = tracking.tracking_filter:main',
            'track_w = tracking.tracking_wave:main',
            'track_3 = tracking.tracking_obb_xyz:main',
            'track_6 = tracking.tracking_obb_6axis:main',
            'track_spin = tracking.tracking_obb_spin_node:main',

        ],
    },
)
