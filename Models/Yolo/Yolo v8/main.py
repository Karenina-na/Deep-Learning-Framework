from ultralytics import YOLO
from PIL import Image
import cv2


def all_format():
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    model = YOLO("yolov8n.pt")
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # results = model.predict(source="0", show=True)
    results = model.predict(source="./inference/images")  # Display preds. Accepts all YOLO predict arguments
    # save


def PIL():
    # from PIL
    model = YOLO("yolov8n.pt")
    im1 = Image.open("bus.jpg")
    results = model.predict(source=im1, save=True)  # save plotted images


def ndarray():
    # from ndarray
    model = YOLO("yolov8n.pt")
    im2 = cv2.imread("bus.jpg")
    results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels


def list():
    # from list of PIL/ndarray
    model = YOLO("yolov8n.pt")
    im1 = Image.open("bus.jpg")
    im2 = cv2.imread("bus.jpg")
    # accepts list of images - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    results = model.predict(source=[im1, im2], save=True, save_txt=True)  # save predictions as labels


def main():
    # results would be a list of Results object including all the predictions by default
    # but be careful as it could occupy a lot memory when there're many images,
    # especially the task is segmentation.
    # 1. return as a list
    model = YOLO("yolov8n.pt")
    results = model.predict(source="inference/images", save=True, save_txt=True)

    # results would be a generator which is more friendly to memory by setting stream=True
    # 2. return as a generator
    # results = model.predict(source=0, stream=True)\
    for result in results:
        # Detection
        print("result boxes xyxy", result.boxes.xyxy)  # box with xyxy format, (N, 4)
        print("result boxes xywh", result.boxes.xywh)  # box with xywh format, (N, 4)
        print("result boxes xyxyn", result.boxes.xyxyn)  # box with xyxy format but normalized, (N, 4)
        print("result boxes xywhn", result.boxes.xywhn)  # box with xywh format but normalized, (N, 4)
        print("result boxes conf", result.boxes.conf)  # confidence score, (N, 1)
        print("result boxes cls", result.boxes.cls)  # cls, (N, 1)

        # Segmentation
        # print("result masks", result.masks)  # masks, (N, H, W)
        # print("result masks xy", result.masks.xy)  # x,y segments (pixels), List[segment] * N
        # print("result masks xyn", result.masks.xyn)  # x,y segments (normalized), List[segment] * N

        # Classification
        print("result probs", result.probs)  # cls prob, (num_class, )

    # Each result is composed of torch.Tensor by default,
    # in which you can easily use following functionality:
    result = result.cuda()
    result = result.cpu()
    result = result.to("cpu")
    result = result.numpy()


if __name__ == "__main__":
    # all_format()
    # PIL()
    # ndarray()
    # list()
    main()
