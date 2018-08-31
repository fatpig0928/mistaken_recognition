import os
import cv2
import time
import numpy as np
from collections import defaultdict

from facedetector import apis


def get_pic_feature(pic_path):
    """识别图片并保存feature(此处需要确保每一张目标图片中只有一张人脸)
    Args:
        pic_path: 图片的地址
    Returns:
        features_pic: 图片的feature[[feature1-1,feature1-2...]],[feature2-1,feature2-2...],[feature3-1,feature3-2...],[feature4-1,feature4-2...]]
        people_names: 每个图片的人名
    """
    features_pic = [] #包含4个文件夹的feature,features_pic的长度等于每个人有的图片数
    people_names = []
    sub_features_pic = []
    imagedir = os.listdir(pic_path) #读取pic_path中所有的文件夹
    for sub_image_dir in imagedir:  #对每一个文件夹识别feature,并保存到sub_features_pic中
        imagelist = os.listdir(os.path.join(pic_path, sub_image_dir))
        for pic_name in imagelist:
            cvimg = cv2.imread(os.path.join(pic_path, sub_image_dir, pic_name))
            feature, bounding_box, landmark = apis.face_feature(cvimg)
            sub_features_pic.append(feature)
            people_names.append(pic_name.split('.')[0])
        features_pic.append(sub_features_pic) #将已经保存文件夹中每一个feature的sub_features_pic添加到features_pic中
        sub_features_pic = []
    people_names = people_names[:len(imagelist)]
    
    return features_pic,people_names


def write_mistaken_face(mistaken_face_path, count,people_name, sim, bb):
    """将误识别的信息输出到txt中
    Args:
        mistaken_face_path: 误识别输出txt的文件名
        count: 帧号
        people_name: 误识别的姓名
        sim: 误识别的sim值
        max_sum: 通过process_sim()函数处理过后的加权分数
        sim_array: 误识别人的对于每张图片的sim值
    """
    if not os.path.exists(mistaken_face_path):
        os.mknod(mistaken_face_path)
    with open(mistaken_face_path, 'a+') as filename:
        filename.write(str(count) + '\t' + people_name + '\t' + str(sim) + '\t'+ str(bb) + '\n')


def rule(index,person_sim,sim_compare):
    """加权处理sim值的规则
    Args:
        index: peron_sim在sim_array中的索引
        person_sim: person_sim即为sim_array[name],当前name与某一个人的4张图片对比feature的sim值[sim1,sim2,sim3,sim4],person_sim的个数是不定的,都是>0的
        sim_compare: 上一次处理后的sim_compare
    Returns:
        sim_compare: 加上处理person_sim的sim_compare
    """
    sim_compare.append(sum(person_sim)) #将sim_array中的每个人的sim值相加

    for sim in person_sim:            
        if sim >= 0.4:
            sim_compare[index] +=3
        elif 0.3 <= sim < 0.4:
            sim_compare[index] +=2
        elif 0.2 <= sim < 0.3:
            sim_compare[index] +=1
        else:
            sim_compare[index] -=1
    
    return sim_compare


def process_sim(sim_array):
    """加权计算sim值
    Args:
        sim_array: 每一帧的每一个人脸与103个人的4张图片比较feature,大于0的保存到sim_array里 {name:[sim1,sim2,sim3,sim4],name:[sim1...]}
    Returns:
        last_name: 最后确定的要把这一帧的这一个人识别成某人的姓名
        last_sim: 最后确定的要把这一帧的这一张人脸识别成某人的sim值
        max_sum: 加权计算后的分数
    """
    name_array = []
    sim_compare = []                            
    for index,name in enumerate(sim_array): #遍历sim_array中的每个人,用于加权计算sim       
        name_array.append(name)
        sim_compare = rule(index,sim_array[name],sim_compare)
    
    #print(sim_compare)
    if sim_compare:
        max_sum = max(sim_compare) 
        if max_sum> 1.2: #3.6是最低标准,及满足每个人的sim值(因为每个人图片的个数为4),满足每个人至少有3张图片被识别成0.2~0.3之间,也就是0.2*3+3
            last_name = name_array[sim_compare.index(max_sum)] #找到加权计算后值最大的人名
            last_sim = np.mean(sim_array[last_name]) #找到加权计算后值最大的人名的sim值并求平均值
            return last_name, last_sim, max_sum 
        else:
            return None,None,None


def draw_box_txt(cvimg, bb, lm, name, sim):
    """给每一帧识别的人脸画框并写名字和sim值
    Args:
        cvimg: 帧
        bb: bounding_box
        lm: landmark
        name: 识别到的人名
        sim: 被识别到人的sim值
    Returns:
        cvimg: 画好框和文字的帧
    """
    cvimg = apis.draw_box(cvimg, bb)
    cvimg = apis.draw_box_text(cvimg, bb, name + str(sim))
    cvimg = apis.draw_landmark(cvimg, lm)  
    return cvimg     


def detect(mistaken_face_path, video_name, features_pic, people_names):
    """识别人脸
    Args:
        mistaken_face_path: 误识别输出txt的文件名
        video_name: 识别的视频名称
        features_pic: 图片的feature,features_pic的长度等于每个人有的图片数
        people_names: 图片的人名
    """
    vidcap = cv2.VideoCapture(video_name)
    success = True
    count = 0
    sim_array = defaultdict(lambda :[])
    multi_frame = []
    
    while success:
        success, cvimg2 = vidcap.read() #每一帧判断

        if success:
            features = apis.face_features(cvimg2) 

            for feature2, bb2, lm2 in features:  #每一帧中的每一个人脸判断 
                for sub_features_pic in features_pic: #每一帧的中的每一个人脸与103个人进行sim计算,大于0的就保存到sim_array中
                    for index, feature in enumerate(sub_features_pic):          
                        sim = apis.feature_sim(feature, feature2)
                        if sim > 0:
                            sim_array[people_names[index]].append(sim)

                last_name,last_sim,max_sum= process_sim(sim_array)
                sim_array = defaultdict(lambda :[]) #sim_array是某一帧中的某一个人脸对应103个人的识别结果,下一个人脸就要重置sim_array
                if last_name:
                    write_mistaken_face(mistaken_face_path, count, last_name, last_sim, bb2) #将误识别写入到文件中
                    multi_frame.append([count, last_name, float(last_sim), list(bb2)])
                    cvimg2 = draw_box_txt(cvimg2, bb2, lm2, last_name, last_sim) 
                     
        cv2.imwrite('test/' + str(count) + '.jpg', cvimg2)
        print('count================================================',count)
        count += 1
    print(multi_frame)
    return multi_frame


def main():
    start_time = time.clock()
    apis.set_gpu(0)

    pic_path = 'picture_more'
    video_name = 'videos/2min.mp4'
    mistaken_face_path = '2min/mistaken_face_1.txt'

    features_pic,people_names = get_pic_feature(pic_path)
    multi_frame = detect(mistaken_face_path, video_name, features_pic, people_names)

    end_time = time.clock()
    print('总用时:',end_time-start_time,'s')


if __name__ == '__main__':
    main()
   


