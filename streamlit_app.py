from io import StringIO
import streamlit as st
from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_MobileNetV1
from PIL import Image
from tog.attacks import *
import os
K.clear_session()

fpath = ''

if __name__ == '__main__':
    
    st.title('YOLOv3对抗样本攻击')


    source = ("图片检测")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                fpath = f'assets/{uploaded_file.name}'
                picture = picture.save(f'assets/{uploaded_file.name}')
                
        else:
            is_valid = False
            
    source_adv = ("无目标攻击", "有目标攻击","消失攻击", "伪造攻击")
    adv_index = st.sidebar.selectbox("选择输入", range(len(source_adv)), format_func=lambda x: source[x])
    
    if is_valid:
        print('valid')
        if st.button('开始检测'):
            weights = 'model_weights/YOLOv3_MobileNetV1.h5'  # 加载训练好的权重
            detector = YOLOv3_MobileNetV1(weights=weights)
            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    input_img = Image.open(fpath)
                    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
                    detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
                    visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes)})

                    st.balloons()
                    
#         if st.button('开始攻击'):
#             visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
#                       'untargeted Adversarial': (x_adv_untargeted, detections_adv_untargeted, detector.model_img_size, detector.classes)})
