def run_the_app():
    # 自己文件上传  -   单文件载入
    st.sidebar.markdown("### 第一步：选择本地的一张图片(png/jpg)...")
    uploaded_file = st.sidebar.file_uploader(" ")
    
    confidence_threshold, overlap_threshold = object_detector_ui()
    
    left_column,middle_column, right_column = st.sidebar.beta_columns(3)
    
    if middle_column.button('检测'):
        image = load_local_image(uploaded_file)
        st.image(uploaded_file, caption='The original image',
                 use_column_width=True)
        
        yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)
        draw_image_with_boxes(image, yolo_boxes, "Real-time Computer Vision",
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (overlap_threshold, confidence_threshold))

@st.cache(show_spinner=False)
def read_markdown(path):
    with open(path, "r",encoding = 'utf-8') as f:  # 打开文件
        data = f.read()  # 读取文件
    return data

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # 1 初始化界面
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(read_markdown("instructions_yolov3.md"))
    
    # 2 下载yolov3的模型文件
    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
        # 下载yolov3文件

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("图像检测参数调节器")   # 侧边栏
    app_mode = st.sidebar.selectbox("切换页面模式:",
        ["Run the app","Show instructions", "Show the source code"])
    
    # 展示栏目三
    if app_mode == "Run the app":
        #readme_text.empty()      # 刷新页面
        st.markdown('---')
        st.markdown('## YOLOv3 检测结果:')
        run_the_app() # 运行内容
    # 展示栏目一
    elif app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    # 展示栏目二
    elif app_mode == "Show the source code":
        readme_text.empty()     # 刷新页面
        st.code(read_markdown("streamlit_app_yolov3.py"))



if __name__ == "__main__":
    file_path = 'yolov3.weights'
    # Path to the Streamlit public S3 bucket
    DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
    
    # External files to download.
    EXTERNAL_DEPENDENCIES = {
        "yolov3.weights": {
            "url": "https://pjreddie.com/media/files/yolov3.weights",
            "size": 248007048
        },
        "yolov3.cfg": {
            "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "size": 8342
        }
    }
    
    
    main()
