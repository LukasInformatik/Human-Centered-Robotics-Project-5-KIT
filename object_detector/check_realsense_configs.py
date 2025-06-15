import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    print("Device:", dev.get_info(rs.camera_info.name))
    for sensor in dev.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.is_video_stream_profile():
                vp = profile.as_video_stream_profile()
                print(vp.stream_type(), vp.width(), vp.height(), vp.fps(), profile.format())