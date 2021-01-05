"""
Implementation of MOSSE for Multi object tracking (MOT)
Combination with YOLO or SSD
"""
import numpy as np
import cv2,os
from utils import _get_img_lists,_linear_mapping,_window_func_2d,_get_gauss_response
from timeit import time
from cftracker.feature import extract_hog_feature,extract_cn_feature


class mosse():
    def __init__(self,learning_rate=0.125,sigma=2.,img_path = 'datasets/surfer/'
                ,bbox_init_gt=[228,118,140,174],chose_ROI=False,num_pretrain=128):
        """
        num_pretrain: Times to generate eight small perturbations (fi) of random affine transformations 
        """
        self.learning_rate=learning_rate
        self.sigma=sigma
        self.frame_list = _get_img_lists(img_path)
        self.frame_list.sort()
        self.first_frame = cv2.imread(self.frame_list[0])
        self.bbox_init_gt=bbox_init_gt
        self.chose_ROI=chose_ROI
        self.num_pretrain=num_pretrain
        self.fps = 0.0

    def init(self):
        # SELECT GROUND TRUTH FOR FIRST FRAME
        if self.chose_ROI:
            bbox = cv2.selectROI('demo', self.first_frame, False, False)
        else:
            bbox = self.bbox_init_gt

        self.first_frame=self._img_processing(self.first_frame)

        # BBOX INITIALIZATION
        x,y,w,h=tuple(bbox)
        self._center=(x+w/2,y+h/2)
        self.w,self.h=w,h
        w,h=int(round(w)),int(round(h))
        
        
        # INITIALIZE Ai AND Bi : TRAINING THE Hi FOR THE FIRST FRAME
        fi_=cv2.getRectSubPix(self.first_frame,(w,h),self._center)

        self._G=np.fft.fft2(_get_gauss_response((w,h),self.sigma))
        self._Ai,self._Bi=self._training(fi_,self._G)

        #TEST hog features extractior
        self.padding = 1.5
        self.cell_size=4
        self.window_size=(int(np.floor(w*(1+self.padding)))//self.cell_size,int(np.floor(h*(1+self.padding)))//self.cell_size)
        x=cv2.resize(fi_,(self.window_size[0]*self.cell_size,self.window_size[1]*self.cell_size))
        x=extract_hog_feature(x, cell_size=self.cell_size)
        print("fhog",x.shape)

    def update(self):
        for idx in range(len(self.frame_list)):
            t1 = time.time()
            current_frame_BGR = cv2.imread(self.frame_list[idx])
            current_frame=self._img_processing(current_frame_BGR)
            
            fi=cv2.getRectSubPix(current_frame,(int(round(self.w)),int(round(self.h))),self._center)
            fi=self._preprocessing(fi)

            Hi=self._Ai/self._Bi
            hi = _linear_mapping(np.real(np.fft.ifft2(Hi)))
            responses=self._detection(Hi,fi)
            gi = responses

            curr=np.unravel_index(np.argmax(gi, axis=None),gi.shape)
            dy,dx=curr[0]-(self.h/2),curr[1]-(self.w/2)
            x_c,y_c=self._center
            x_c+=dx
            y_c+=dy
            self._center=(x_c,y_c)
            fi=cv2.getRectSubPix(current_frame,(int(round(self.w)),int(round(self.h))),self._center)
            fi=self._preprocessing(fi)

            # UPDATE
            Fi=np.fft.fft2(fi)
            self._Ai=self.learning_rate*(self._G*np.conj(Fi))+(1-self.learning_rate)*self._Ai 
            self._Bi=self.learning_rate*(Fi*np.conj(Fi))+(1-self.learning_rate)*self._Bi
            
            # visualize the tracking process...
            self.fps = (self.fps + (1. / (time.time() - t1))) / 2
            cv2.putText(current_frame_BGR, "FPS:" + str(round(self.fps, 1)), (5, 25), 2, 5e-3 * 200, (0, 255, 0), 2)
            
            cv2.rectangle(current_frame_BGR, (int(self._center[0]-self.w/2),int(self._center[1]-self.h/2)), (int(self._center[0]+self.w/2),int(self._center[1]+self.h/2)), (255, 0, 0), 2)
            cv2.imshow('demo', current_frame_BGR)
            cv2.waitKey(1)
            # cv2.imshow('fi',fi)
            # cv2.waitKey(1)
            # cv2.imshow('gi',gi)
            # cv2.waitKey(1)
            # cv2.imshow('hi',hi)
            # cv2.waitKey(1)
    
    def _img_processing(self,img):
        if len(img.shape)!=2:
            assert img.shape[2]==3
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)/255

    def _training(self,fi,G):
        Ai, Bi= np.zeros_like(G),np.zeros_like(G)
        for _ in range(self.num_pretrain):
            # AFFINE TRANSFORMATION 
            fi=self._rand_warp(fi)
            Fi=np.fft.fft2(self._preprocessing(fi))
            Ai+=G*np.conj(Fi)
            Bi+=Fi*np.conj(Fi)
        Hi = Ai/Bi
        return Ai,Bi

    def _detection(self,Hi,fi):
        Gi=Hi*np.fft.fft2(fi)
        gi = _linear_mapping(np.real(np.fft.ifft2(Gi)))
        return gi

    def _preprocessing(self,img,eps=1e-5):
        # First, the pixel values are transformed using a log function which helps with low contrast lighting situations. 
        img=np.log(img+1)
        # The pixel values are normalized to have a mean value of 0.0 and a norm of 1.0
        img=(img-np.mean(img))/(np.std(img)+eps)
        # Finally, the image is multiplied by a cosine window which gradually
        # reduces the pixel values near the edge to zero. This also
        # has the benefit that it puts more emphasis near the center of the target.
        cos_window = _window_func_2d(int(round(self.w)),int(round(self.h)))
        preprocessed_img = cos_window*img
        return preprocessed_img

    def _rand_warp(self,img):
        """
        The training set is constructed using
        random affine transformations to generate eight small perturbations (fi)
        of the tracking window in the initial frame.
        """
        h, w = img.shape[:2]
        C = .1
        ang = np.random.uniform(-C, C)
        c, s = np.cos(ang), np.sin(ang)
        W = np.array([[c + np.random.uniform(-C, C), -s + np.random.uniform(-C, C), 0],
                      [s + np.random.uniform(-C, C), c + np.random.uniform(-C, C), 0]])
        center_warp = np.array([[w / 2], [h / 2]])
        tmp = np.sum(W[:, :2], axis=1).reshape((2, 1))
        W[:, 2:] = center_warp - center_warp * tmp
        warped = cv2.warpAffine(img, W, (w, h), cv2.BORDER_REFLECT)
        return warped


if __name__ == "__main__":

    init_gt=[228,118,140,174]
    img_path = 'datasets/surfer/'

    tracker = mosse(img_path=img_path,bbox_init_gt=init_gt,chose_ROI=False)
    tracker.init()
    tracker.update()
