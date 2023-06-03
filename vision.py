import cv2
import sys
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

##################################
# Some default variables
##################################
scale = 0.5 # How much to resize the image for detection
image_max_height = (3456 * scale)
image_midpoint_width = (4608 * scale) / 2


# Unwrap the X, Y, coordinates from a bounding box
def __get_xy(box):
    center = box.get_centers()[0]
    x = center[0]
    y = center[1]
    return x, y

# Get the distances (in pixels) from launch position
# negative indicates left, positive indicates right
def __get_pixel_dist(box):
    x, y = __get_xy(box)
    horz = x - image_midpoint_width
    vert = y
    return horz, vert


# Calculate the distance from the 'starting position' for a bounding box.
# Starting position reference is horizontal middle and vertical bottom
def __calc_proximity(box):
    horz, vert = __get_pixel_dist(box)
    return pow(pow(abs(horz), 2) + pow(abs(vert), 2), 0.5)


# Returns the X, Y, and distance to the closest identified parking spot.
def find_closest_spot(image_path):
    ##################################
    # Load trained model and set up
    ##################################
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    register_coco_instances("my_dataset_train", {}, "/home/kingclark/ase/uas-4-mls/images/train/coco labels.json",
                            "/home/kingclark/ase/uas-4-mls/images/train/")
    cfg.MODEL.WEIGHTS = "/home/kingclark/ase/uas-4-mls/vision/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_train")

    ##################################
    # For now; pick an image and calculate predictions
    ##################################
    im = cv2.imread(image_path)
    outputs = predictor(
        im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    ##################################
    # Pick our "closest" prediction
    ##################################

    closest_point = -1
    closest_dist = sys.maxsize
    for idx, val in enumerate(outputs['instances'].get_fields()['pred_classes']):
        if val == 0:
            box = outputs['instances'].get_fields()['pred_boxes'][idx]
            dist = __calc_proximity(box)
            if dist < closest_dist:
                closest_dist = dist
                closest_point = idx
    return (__get_pixel_dist(outputs['instances'].get_fields()['pred_boxes'][closest_point]), closest_dist)

    # ##################################
    # # Debug print images, put above the return to see bounding boxes on image
    # ##################################
    # from detectron2.utils.visualizer import Visualizer
    # v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=scale, )
    # v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.namedWindow("dummy", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("dummy", 1920, 1080)
    # cv2.imshow("dummy", out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# ##################################
# # Test, run on a random image
# ##################################
# import random
# import glob
# image = random.sample(glob.glob("/home/kingclark/ase/uas-4-mls/images/test/*JPG"), 1)[0]
# print(f'Found spot at: {find_closest_spot(image)} for {image}')
