import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'project_5_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools',
                    'opencv-python',
                    'ultralytics',
                    'numpy<2',
                    'mediapipe',
                    'ahrs',
                    'tf-transformations'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='lukas.d.ringle@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_person_tracker_node = project_5_pkg.yolo_person_tracker_node:main',
            'human_localizer_node = project_5_pkg.human_localizer_node:main',
            'camera_orientation_node = project_5_pkg.camera_orientation_node:main'
        ],
    },
)
