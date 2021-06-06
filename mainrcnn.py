from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.utils import Dataset
from numpy import zeros
from numpy import asarray
from os import listdir
from xml.etree import ElementTree

class myMaskRCNNConfig(Config):
    NAME = "MaskRCNN_config"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 6

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Learning rate
    LEARNING_RATE = 0.006

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # setting Max ground truth instances
    MAX_GT_INSTANCES = 10

config = myMaskRCNNConfig()
config.display()


class PIDDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, images_dir,annotations_dir,is_train=True):
        self.add_class("dataset", 1, "Country")
        self.add_class("dataset", 2, "Name")
        self.add_class("dataset", 3, "Identity Card No")
        self.add_class("dataset", 4, "Date of Birth")
        self.add_class("dataset", 5, "Date of Expiry")

        for filename in listdir(images_dir):

            # extract image id
            image_id = filename[:-4]
            # skip bad images
            if image_id in ['00090']:
                continue
            if is_train and int(image_id) >= 1820:
                continue
            if not is_train and int(image_id) < 1820:
                continue

            # setting image file
            img_path = images_dir + filename

            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3,4,5])

    def extract_boxes(self, filename):

        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
            if name=='Date of Expiry' or name=='Date of Birth' or name=='Name' or name=='Country' or name=='Identity Card No':
                boxes.append(coors)

        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]

        # define anntation  file location
        path = info['annotation']

        # load XML
        boxes, w, h = self.extract_boxes(path)

        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            if (box[4] == 'Country'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('Country'))
            if (box[4] == 'Name'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('Name'))
            if (box[4] == 'Identity Card No'):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('Identity Card No'))
            if (box[4] == 'Date of Birth'):
                masks[row_s:row_e, col_s:col_e, i] = 4
                class_ids.append(self.class_names.index('Date of Birth'))
            if (box[4] == 'Date of Expiry'):
                masks[row_s:row_e, col_s:col_e, i] = 5
                class_ids.append(self.class_names.index('Date of Expiry'))
        return masks, asarray(class_ids, dtype='int32')

    """Return the path of the image."""

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']


train_set = PIDDataset()

print ("Loading training data")
dataset_dir="C:\\Users\\abdul\\PycharmProjects\\DSProj1\\Mask_RCNN\\"
images_dir = dataset_dir + 'Pakistan CNIC\\'
annotations_dir = dataset_dir + 'Annotated-Pakistan CNIC-YOLO\\'
train_set.load_dataset(images_dir,annotations_dir, is_train=True)
print ("Loaded training data")
train_set.prepare()
print("Train: %d" % len(train_set.image_ids))

# prepare test/val set
test_set = PIDDataset()
test_set.load_dataset(images_dir,annotations_dir, is_train=False)
test_set.prepare()
print("Test: %d" % len(test_set.image_ids))

print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')
model.load_weights('C:\\Users\\abdul\\PycharmProjects\\DSProj1\\Mask_RCNN\\mask_rcnn_coco.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_set, test_set, learning_rate=2*config.LEARNING_RATE, epochs=1, layers="heads")
history = model.keras_model.history.history

import time
model_path = 'C:\\Users\\abdul\\PycharmProjects\\DSProj1\\mask_rcnn_'  + '.' + str(time.time()) + '.h5'
model.keras_model.save_weights(model_path)

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#Loading the model in the inference mode

model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
# loading the trained weights o the custom dataset
model.load_weights(model_path, by_name=True)
img = load_img("C:\\Users\\abdul\\PycharmProjects\\DSProj1\\Mask_RCNN\Pakistan CNIC\\20.jpg")
img = img_to_array(img)
# detecting objects in the image
result= model.detect([img])

image_id = 1
image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(test_set, config, image_id, use_mini_mask=False)
info = test_set.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       test_set.image_reference(image_id)))
# Run object detection
results = model.detect([image], verbose=1)
# Display results

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            test_set.class_names, r['scores'],
                            title="Predictions")


train_set = PIDDataset()

print ("Loading training data")
dataset_dir="C:\\Users\\abdul\\PycharmProjects\\DSProj1\\Mask_RCNN\\"
images_dir = dataset_dir + 'Uk passport\\'
annotations_dir = dataset_dir + 'Annotated-UK Passport-YOLO\\'

train_set.load_dataset(images_dir,annotations_dir, is_train=True)
print ("Loaded training data")
train_set.prepare()
print("Train: %d" % len(train_set.image_ids))

# prepare test/val set
test_set = PIDDataset()
test_set.load_dataset(images_dir,annotations_dir, is_train=False)
test_set.prepare()
print("Test: %d" % len(test_set.image_ids))

print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')
model.load_weights('C:\\Users\\abdul\\PycharmProjects\\DSProj1\\Mask_RCNN\\mask_rcnn_coco.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_set, test_set, learning_rate=2*config.LEARNING_RATE, epochs=1, layers="heads")
history = model.keras_model.history.history

import time
model_path = 'C:\\Users\\abdul\\PycharmProjects\\DSProj1\\mask_rcnn_'  + '.' + str(time.time()) + '.h5'
model.keras_model.save_weights(model_path)

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#Loading the model in the inference mode

model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
# loading the trained weights o the custom dataset
model.load_weights(model_path, by_name=True)
img = load_img("C:\\Users\\abdul\\PycharmProjects\\DSProj1\\Mask_RCNN\\Uk passport\\20.jpg")
img = img_to_array(img)
# detecting objects in the image
result= model.detect([img])

image_id = 1
image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(test_set, config, image_id, use_mini_mask=False)
info = test_set.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       test_set.image_reference(image_id)))
# Run object detection
results = model.detect([image], verbose=1)
# Display results

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            test_set.class_names, r['scores'],
                            title="Predictions")