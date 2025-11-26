import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V81 - Kara Kutu)", page_icon="⬛", layout="wide")

# --- YAPAY ZEKA MOTORU ---
@st.cache_resource
def load_model():
    return easyocr.Reader(['tr', 'en'], gpu=False)

try:
    reader = load_model()
except Exception as e:
    st.error("Model Yüklenemedi.")
    st.stop()

# --- GÖRÜNTÜ İŞLEME ---
def resmi_hazirla(pil_image):
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return gray

# --- SAYI TEMİZLEME ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
        t = t.replace('O', '0').replace('S', '5').replace('I', '1').replace('L', '1').replace('Z', '2').replace('B', '8')
        if "3/0" in t: t = t.replace("3/0", "370")
        
        t = t.replace(' ', '').replace('*', '').replace('TL', '')
        t = re.sub(r'[^\d,.]', '', t)
        
        if len(t) > 0:
            t = t.replace('.', 'X').replace(',', '.').replace('X', '')
            return float(t)
    except:
        pass
    return 0.0

# --- 🧬 SATIR BİRLEŞTİRİCİ 🧬 ---
def grupla_ve_satir_yap(ocr_results, y_tolerans=15):
    # Y koordinatına göre sırala
    sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
    
    satirlar = []
    if not sorted_results: return satirlar

    mevcut_satir = [sorted_results[0]]
    mevcut_y = sorted_results[0][0][0][1]

    for i in range(1, len(sorted_results)):
        box, text, conf = sorted_results[i]
        y = box[0][1]

        if abs(y - mevcut_y) < y_tolerans:
            mevcut_satir.append(sorted_results[i])
        else:
            mevcut_satir.sort(key=lambda x: x[0][0][0])
            satir_metni = " ".join([item[1] for item in mevcut_satir])
            satirlar.append(satir_metni)
            
            mevcut_satir = [sorted_results[i]]
            mevcut_y = y
            
    if mevcut_satir:
        mevcut_satir.sort(key=lambda x: x[0][0][0])
        satirlar.append(" ".join([item[1] for item in mevcut_satir]))
        
    return satirlar

# --- ANALİZ MOTORU ---
def veri_analiz
