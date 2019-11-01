import sys
import dlib
import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume",type=str,required=True,help="your facial shape predictor of 5 face landmarks")
    parser.add_argument("--face_path",type=str,required=True,help="the face img to dectect")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    detector = dlib.get_frontal_face_detector()  #人脸检测器，用于寻找每一张脸的bounding boxes
    sp = dlib.shape_predictor(args.resume)       #调用sp返回一个full_object_detections的对象，sp表示特征点预测
    img = dlib.load_rgb_image(args.face_path)

    dets = detector(img,1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry,there were no faces found in '{}'".format(args.face_path))
        exit(-1)
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img,detection))   #调用sp 预测检测到人脸的特征点
    window = dlib.image_window()
    images = dlib.get_face_chips(img,faces,size=320)   #把在img中的经过校正的faces返回到images
    for image in images:
        window.set_image(image)
        dlib.hit_enter_to_continue()

if __name__ == '__main__':
    main()

