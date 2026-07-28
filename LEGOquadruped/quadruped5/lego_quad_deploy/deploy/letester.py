import legoeducation as le, time
dm = le.DoubleMotor();dm.connect(card_color=le.LEGO_COLOR_PURPLE, card_serial="6040")
dm.imu_set_yaw_face(yaw_face = le.DEVICE_FACE_BACK)

while True:
    time.sleep(0.1)
    print(f"Yaw: {dm.imu_device.yaw} | Pitch: {dm.imu_device.pitch} | Roll: {dm.imu_device.roll}")
# dm.device_notification_request(50)   # push updates every 50 ms
# def cb(data):
#     for item in le.device_notification_parser(data):
#         print(type(item).__name__, vars(item))
# dm.set_notification_callback(cb)
# time.sleep(3); dm.disconnect()