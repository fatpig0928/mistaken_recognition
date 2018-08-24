import os
import cv2
from facedetector import apis
import time
import numpy as np


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

def write_mistaken_face(mistaken_face_path,video_name,people_name,sim,count):
    if not os.path.exists(mistaken_face_path):
        os.mknod(mistaken_face_path)
    with open(mistaken_face_path, 'a+') as filename:
        filename.write(video_name + ' ' + people_name + ' ' + str(sim) + ' '+ str(count) + ' ' + '\n')
    
def detect(mistaken_face_path,video_name,features_pic,people_names,features_pic_v2,features_pic_v3,features_pic_v4):
    vidcap = cv2.VideoCapture(video_name)
    success = True
    count = 0
    #分帧并保存jpg方法一()
    while success:

        #跳帧读取
        # frame_freq = 1
        # for i in range(frame_freq):
        #     success, cvimg2 = vidcap.read()
        #     if not i == frame_freq - 1:
        #         vidwrt.write(cvimg2)

        success, cvimg2 = vidcap.read()

        if success:
            features = apis.face_features(cvimg2)
            for feature2, bb2, lm2 in features:
                sim_array = []
                name_array = []
                for index, feature in enumerate(features_pic):
                    sim = apis.feature_sim(feature, feature2)
                    sim_array.append(sim)
                    name_array.append(people_names[index])
                max_sim = max(sim_array)
                ind = sim_array.index(max_sim)
                v2_index = people_names.index(name_array[ind])
                if max_sim >= 0.2:
                    write_mistaken_face(mistaken_face_path[0],video_name,people_names[v2_index],max_sim,count)
                    sim_v2 = apis.feature_sim(features_pic_v2[v2_index], feature2)
                    if sim_v2 >= 0.2:
                        write_mistaken_face(mistaken_face_path[1],video_name,people_names[v2_index],np.mean([max_sim,sim_v2]),count)
                        sim_v3 = apis.feature_sim(features_pic_v3[v2_index], feature2)
                        if sim_v3 >= 0.2:
                            write_mistaken_face(mistaken_face_path[2],video_name,people_names[v2_index],np.mean([max_sim,sim_v2,sim_v3]),count)
                            sim_v4 = apis.feature_sim(features_pic_v4[v2_index], feature2)
                            if sim_v4 >= 0.2:
                                write_mistaken_face(mistaken_face_path[3],video_name,people_names[v2_index],np.mean([max_sim,sim_v2,sim_v3,sim_v4]),count)
                                cvimg2 = apis.draw_box(cvimg2, bb2)
                                cvimg2 = apis.draw_box_text(cvimg2, bb2, people_names[v2_index]+str(np.mean([max_sim,sim_v2,sim_v3,sim_v4])))
                                cvimg2 = apis.draw_landmark(cvimg2, lm2)
        cv2.imwrite('test/' + str(count) + '.jpg', cvimg2)
        print('count================================================',count)
        count += 1

if __name__ == '__main__':

    start_time = time.clock()
    apis.set_gpu(0)
    pic_path = '1/falling_Officer'
    pic_path_v2 = '1/falling_Officer_v2'
    pic_path_v3 = '1/falling_Officer_v3'
    pic_path_v4 = '1/falling_Officer_v4'

    video_name = 'videos/V1533884176498.mp4'

    mistaken_face_path=[]
    mistaken_face_path.append('2min/mistaken_face_v1.txt')
    mistaken_face_path.append('2min/mistaken_face_v2.txt')
    mistaken_face_path.append('2min/mistaken_face_v3.txt')
    mistaken_face_path.append('2min/mistaken_face_v4.txt')

    features_pic,people_names = get_pic_feature(pic_path)
    features_pic_v2,people_names_v2 = get_pic_feature(pic_path_v2)
    features_pic_v3,people_names_v3 = get_pic_feature(pic_path_v3)
    features_pic_v4,people_names_v4 = get_pic_feature(pic_path_v4)

    detect(mistaken_face_path,video_name,features_pic,people_names,features_pic_v2,features_pic_v3,features_pic_v4)
    end_time = time.clock()
    print('总用时:',end_time-start_time,'s')


