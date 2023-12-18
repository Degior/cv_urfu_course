import cv2 as cv
import numpy as np


def blue_mask(frame):
    lower_blue = np.array([90, 70, 70])
    upper_blue = np.array([150, 255, 255])
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    return mask


def remove_mask_noise(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask_erosion = cv.erode(mask, kernel)
    mask_dilation = cv.dilate(mask_erosion, kernel)
    return mask_dilation


video_capture = cv.VideoCapture(".\\input.mp4")

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 60, (1920, 540))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    resized_frame = cv.resize(frame, (960, 540))
    blur_frame = cv.GaussianBlur(resized_frame, (5, 5), 0)
    mask_frame = blue_mask(blur_frame)
    mask_frame = remove_mask_noise(mask_frame)
    contours, _ = cv.findContours(mask_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    gray_frame = cv.cvtColor(mask_frame, cv.COLOR_GRAY2RGB)
    for contour in contours:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        cv.drawContours(blur_frame, [box], 0, (100, 200, 100), 3)
        cv.drawContours(gray_frame, [box], 0, (100, 200, 100), 3)

    merged_video = np.concatenate((blur_frame, gray_frame), axis=1)
    out.write(merged_video)
    cv.imshow('video', merged_video)
    if cv.waitKey(16) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv.destroyAllWindows()
