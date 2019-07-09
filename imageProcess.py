import cv2
import random
import numpy as np
class Img_process:
    def __init__(self):
        pass
    #截取长为lenth,宽为width的图片
    def img_crop(self,ori_img,lenth,width):
        if ori_img is None:
            return None
        if lenth > ori_img.shape[0] or width > ori_img.shape[1]:
            return ori_img
        return ori_img[0:lenth,0:width]
    def color_split(self,ori_img): #返回的是B,G,R的x*y*1的灰度B,G,R
        B,G,R = cv2.split(ori_img)
        return B,G,R
    def color_shift(self,ori_img):
        #先分离出B,G,R三色
        B,G,R = cv2.split(ori_img)
        print(B.shape)
        tri_color = [B,G,R]
        for i in range(len(tri_color)):
            rand_i = random.randint(-50,50)
            color = tri_color[i]
            if rand_i == 0:
                pass
            elif rand_i>0:
                lim = 255 - rand_i
                color[color>lim] = 255
                color[color<=lim] = (rand_i+color[color<=lim]).astype(img.dtype)
            else:
                lim = 0 - rand_i
                color[color < lim] = 0
                color[color >= lim] = (rand_i + color[color>=lim]).astype(img.dtype)
        img_merge = cv2.merge((B,G,R))
        return img_merge

#将原始图像ori_img，按照原中心旋转到目的中心，并按照角度和比例来旋转
 #中心点的坐标为(长，宽），故表示为（列/2，行/2）
    def rotation(self,ori_img,center,angle,scale):
        M = cv2.getRotationMatrix2D(center,angle,scale)
        img_rotate = cv2.warpAffine(ori_img,M,(ori_img.shape[1],ori_img.shape[0]))
        return img_rotate
    #对img做仿射变换,需要3个点来对应图像前后点的关系（目前在函数内部写死，后期可以在函数内部改）
    def affine_transform(self,img):
        rows,cols,ch = img.shape
        pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1]])
        pts2 = np.float32([[cols * 0.2,rows * 0.1],[cols*0.9,rows*0.2],[cols*0.1,rows*0.9]])
        M = cv2.getAffineTransform(pts1,pts2)
        return cv2.warpAffine(img,M,(cols,rows))

    #做perspective_transform后，平行线可能不平行了，直角也可能不是直角了
    #需要找到4个点，表示前后图像的对应，本例中为随机生成4个相对的点
    def perspective_transform(self,img):
        height, width, channels = img.shape
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        return img_warp
    def test(self,img):
        img_crop = self.img_crop(img,150,150)
        img_color_shift = self.color_shift(img)
        img_rotate = self.rotation(img,(img.shape[1]/3,img.shape[0]/3),45,1)
        img_affine_transform = self.affine_transform(img)
        img_perspective_transform = self.perspective_transform(img)
        cv2.imshow('crop',img_crop)
        cv2.imshow('color_shift',img_color_shift)
        cv2.imshow('rotate',img_rotate)
        cv2.imshow('affine transform',img_affine_transform)
        cv2.imshow('perspective transform',img_perspective_transform)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()