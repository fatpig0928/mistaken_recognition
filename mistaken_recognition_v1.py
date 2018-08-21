import os
import cv2
from facedetector import apis
import time


#获取图片的Feature
def get_pic_feature(pic_path):
    features_pic = []
    people_names = []
    imagelist = os.listdir(pic_path)
    for pic_name in imagelist:
        cvimg = cv2.imread(os.path.join(pic_path,pic_name))
        feature, bounding_box, landmark = apis.face_feature(cvimg)
        features_pic.append(feature)
        people_names.append(pic_name.split('.')[0])
    
    return features_pic,people_names

def write_mistaken_face(video_name,people_name,sim,count):
    mistaken_face_path = 'mistaken_face.txt'
    if not os.path.exists(mistaken_face_path):
        os.mknod(mistaken_face_path)
    with open(mistaken_face_path, 'a+') as filename:
        filename.write(video_name + ' ' + people_name + ' ' + str(sim) + ' '+ str(count) + ' ' + '\n')
    
def detect(video_name,features_pic,people_names):
    vidcap = cv2.VideoCapture(video_name)
    success = True
    count = 0
    #分帧并保存jpg方法一()
    while success:

        frame_freq = 1
        for i in range(frame_freq):
            success, cvimg2 = vidcap.read()
            if not i == frame_freq - 1:
                vidwrt.write(cvimg2)

        if success:
            features = apis.face_features(cvimg2)
            for feature2, bb2, lm2 in features:
                sim_array = []
                name_array = []
                for index, feature in enumerate(features_pic):
                    sim = apis.feature_sim(feature, feature2)
                    sim_array.append(sim)
                    name_array.append(people_names[index])
                    #如果相似指数大于0.3, 则认为为同一个人, 画出框框
                max_sim = max(sim_array)
                ind = sim_array.index(max_sim)
                if max_sim >= 0.30:
                    write_mistaken_face(video_name,name_array[ind],max_sim,count)
                    cvimg2 = apis.draw_box(cvimg2, bb2)
                    cvimg2 = apis.draw_box_text(cvimg2, bb2, name_array[ind]+str(max_sim))
                    cvimg2 = apis.draw_landmark(cvimg2, lm2)

        cv2.imwrite('test/' + str(count) + '.jpg', cvimg2)
        print('count================================================',count)
        count += 1

if __name__ == '__main__':

    start_time = time.clock()
    apis.set_gpu(0)
    pic_path = 'falling Officer'
    video_name = '20180630-221715_20180701-001457.mp4'
    features_pic,people_names = get_pic_feature(pic_path)
    detect(video_name,features_pic,people_names)
    end_time = time.clock()
    print('总用时:',end_time-start_time,'s')


