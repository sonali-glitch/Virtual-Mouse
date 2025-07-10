import cv2

def rescale_frame(frame, scale=1.0):
    """
    Rescales the given frame to the specified scale.
    
    :param frame: The original image/frame to be resized.
    :param scale: The scaling factor. A value > 1.0 will zoom in, while a value < 1.0 will zoom out.
    :return: The resized frame.
    """
    height, width = frame.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    resized_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_LINEAR)
    return resized_frame

def main():
    cap = cv2.VideoCapture(0)  # Capture video from webcam
    scale = 1.0  # Initial scale factor

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rescale the frame
        frame_rescaled = rescale_frame(frame, scale)

        # Display the rescaled frame
        cv2.imshow('Zoom In/Out', frame_rescaled)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):  # Press '+' key to zoom in
            scale += 0.1
        elif key == ord('-') and scale > 0.1:  # Press '-' key to zoom out
            scale -= 0.1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
