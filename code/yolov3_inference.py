import cv2
import numpy as np
from scipy.special import expit
from keras.models import load_model

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

class YoloPredictor:
    def __init__(self, model_path):
        self.infer_model = load_model(model_path)
        self.net_h = 224
        self.net_w = 352
        self.obj_thresh = 0.5
        self.nms_thresh = 0.45
        self.anchors = [88,47, 91,36, 92,55, 92,28, 93,69, 93,41, 96,61, 97,81, 97,48]

    def predict(self, input_image):
        image = cv2.imread(input_image)
        boxes = self.get_yolo_boxes(image)
        return boxes

    def get_yolo_boxes(self, image):
        processed_image = self.preprocess_input(image)
        output = self.infer_model.predict(processed_image)
        yolos = [output[0][0], output[1][0], output[2][0]]
        boxes = []
        for j in range(len(yolos)):
            yolo_anchors = self.anchors[(2-j)*6:(3-j)*6]
            boxes += self.decode_netout(yolos[j], yolo_anchors)

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
    
    def decode_netout(self, netout, anchors):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5

        boxes = []

        netout[..., :2]  = self.sigmoid(netout[..., :2])
        netout[..., 4]   = self.sigmoid(netout[..., 4])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * self.softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > self.obj_thresh

        for i in range(grid_h*grid_w):
            row = i // grid_w
            col = i % grid_w
            
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[row, col, b, 4]
                
                if(objectness <= self.obj_thresh): continue
                
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[row,col,b,:4]

                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / self.net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / self.net_h # unit: image height  
                
                # last elements are class probabilities
                classes = netout[row,col,b,5:]
                
                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

                boxes.append(box)

        return boxes
        
    def sigmoid(self, x):
        return expit(x)
    
    def softmax(self, x, axis=-1):
        x = x - np.amax(x, axis, keepdims=True)
        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)

if __name__ == '__main__':
    path = 'images\AfricaDrone_Alt0.2_Deg0_0_1582957164.png'
    model_path = '..\weights\drone.h5'
    yp = YoloPredictor(model_path)
    output = yp.predict(path)
    print(f'output = {output.shape}')
    print(len(output))