'''
video_test.py utilizes video stream to test various kernel under development
'''

# import essentials
import cv2
import line_kernel

# Define hyperparameters here
VID_WIDTH = 640
VID_HEIGHT = 480
VID_FPS = 30


def main():
    # setup camera
    vid = cv2.VideoCapture(0)       # set the video mode
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, VID_WIDTH)       # decide the size of the live stream
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_HEIGHT)
    vid.set(cv2.CAP_PROP_FPS, VID_FPS)        # framerate

    while True:
        _, frame = vid.read()      # read out the frame
        _,_,output_bundle = line_kernel.process(frame)

        # show the output
        cv2.imshow('output', output_bundle[0])
        if cv2.waitKey(1) & 0xFF is ord('q'):       # waiting for the user to quit
            break
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
