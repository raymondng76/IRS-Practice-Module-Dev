import cv2
import numpy as np
from keras.models import load_model

class YoloPredictor:
    def __init__(self, model_path):
        self.infer_model = load_model(model_path)
        self.net_h = 224
        self.net_w = 352
        self.obj_thresh = 0.5
        self.nms_thresh = 0.45

    def predict(self, input_image):
        image = cv2.imread(input_image)
        print(image.shape)
        boxes = self.get_yolo_boxes(image)


    def get_yolo_boxes(self, image):
        processed_image = self.preprocess_input(image)
        output = self.infer_model.predict(processed_image)
        print(output.shape)
        return output

    def preprocess_input(self, image):
        new_h, new_w, _ = image.shape

        # determine the new size of the image
        if (float(self.net_w)/new_w) < (float(self.net_h)/new_h):
            new_h = (new_h * self.net_w)//new_w
            new_w = self.net_w
        else:
            new_w = (new_w * self.net_h)//new_h
            new_h = self.net_h

        # resize the image to the new size
        resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

        # embed the image into the standard letter box
        new_image = np.ones((self.net_h, self.net_w, 3)) * 0.5
        new_image[(self.net_h-new_h)//2:(self.net_h+new_h)//2, (self.net_w-new_w)//2:(self.net_w+new_w)//2, :] = resized
        new_image = np.expand_dims(new_image, 0)

        return new_image

if __name__ == '__main__':
    path = 'code\images\AfricaDrone_Alt0.2_Deg0_0_1582957164.png'
    model_path = '..\weights\drone.h5'
    yp = YoloPredictor(model_path)
    output = yp.predict(path)
    