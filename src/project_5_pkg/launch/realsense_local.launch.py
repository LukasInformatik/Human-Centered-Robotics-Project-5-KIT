from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('project_5_pkg')
    yaml_file = os.path.join(pkg_share, 'config', 'camera_params.yaml')

    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='d435i',
            namespace='go2',
            parameters=[yaml_file],
            output='screen',
            emulate_tty=True,
        )
    ])
