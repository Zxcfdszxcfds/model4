import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="图像处理简易平台", layout="wide")
st.title("📷 简易图像处理实验平台")

# 上传图片
img_file = st.file_uploader("上传图片", type=["jpg","png","jpeg"], key="up1")

if img_file:
    img = Image.open(img_file).convert("RGB")
    img_np = np.array(img)
    st.image(img, caption="原图", use_column_width=True)

    # 1. 图像灰度化
    st.subheader("1. 图像灰度化")
    if st.button("转为灰度图", key="gray_btn"):
        gray = np.dot(img_np[...,:3], [0.299,0.587,0.114])
        fig,ax = plt.subplots()
        ax.imshow(gray, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    # 2. 图像二值化
    st.subheader("2. 图像二值化")
    thres = st.slider("阈值", 0, 255, 127, key="thr_slide")
    if st.button("执行二值化", key="bin_btn"):
        gray = np.dot(img_np[...,:3], [0.299,0.587,0.114])
        binary = np.where(gray>thres,255,0)
        fig,ax = plt.subplots()
        ax.imshow(binary, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    # 3. 边缘检测（修复版）
    st.subheader("3. 简易边缘提取")
    if st.button("提取边缘", key="edge_btn"):
        gray = np.dot(img_np[...,:3], [0.299,0.587,0.114])
        # 修复：分别对行和列做diff后，统一补回形状再相加
        dx = np.abs(np.diff(gray, axis=1, prepend=gray[:,0:1]))
        dy = np.abs(np.diff(gray, axis=0, prepend=gray[0:1,:]))
        edge = dx + dy
        fig,ax = plt.subplots()
        ax.imshow(edge, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

st.markdown("---")
st.caption("轻量版图像处理作业 | 无依赖冲突 · 云端秒跑")
