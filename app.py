import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import re
import io
import cv2
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V102 - Hibrid)", page_icon="🪂", layout="wide")

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    if not os.path.exists("best.pt"):
        st.error("⚠️ 'best.pt' bulunamadı!")
        st.stop()
    
    detector = YOLO('best.pt')
    reader = PaddleOCR(use_angle_cls=True, lang='tr')
    return detector, reader

try:
    detector, reader = load_models()
except Exception as e:
    st.error(f"Hata: {e}")
    st.stop()

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

# --- 1. YÖNTEM: KLASİK ANALİZ (YEDEK PARAŞÜT) ---
def paddle_sonuclari_duzenle(results):
    if not results or results[0] is None: return []
    sorted_res = sorted(results[0], key=lambda x: x[0][0][1])
    satirlar = []
    if not sorted_res: return satirlar

    mevcut_satir = [sorted_res[0]]
    mevcut_y = sorted_res[0][0][0][1]

    for i in range(1, len(sorted_res)):
        box = sorted_res[i][0]
        y = box[0][1]
        if abs(y - mevcut_y) < 15:
            mevcut_satir.append(sorted_res[i])
        else:
            mevcut_satir.sort(key=lambda x: x[0][0][0])
            text_line = " ".join([item[1][0] for item in mevcut_satir])
            satirlar.append(text_line)
            mevcut_satir = [sorted_res[i]]
            mevcut_y = y
            
    if mevcut_satir:
        mevcut_satir.sort(key=lambda x: x[0][0][0])
        satirlar.append(" ".join([item[1][0] for item in mevcut_satir]))
    return satirlar

def klasik_analiz(satirlar):
    veriler = {'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0}
    full_text = " ".join(satirlar).upper()
    
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    zno = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno: veriler['Z_No'] = zno.group(1)

    for i, s in enumerate(satirlar):
        s_upper = s.upper()
        if "KUM" in s_upper or "KÜM" in s_upper: continue
        adaylar = re.findall(r'[\d\.,]+', s_upper)
        
        if "NAKİT" in s_upper or "NAKIT" in s_upper:
            for a in adaylar:
                v = sayi_temizle(a)
                if v > 0 and v < 500000:
                    if not (v < 50 and float(v).is_integer()): veriler['Nakit'] = max(veriler['Nakit'], v)
            if i+1 < len(satirlar): # Alt satır
                for a in re.findall(r'[\d\.,]+', satirlar[i+1]):
                    v = sayi_temizle(a)
                    if v > 0 and v < 500000 and not (v < 50 and float(v).is_integer()): veriler['Nakit'] = max(veriler['Nakit'], v)

        if ("KREDİ" in s_upper or "KART" in s_upper) and "YEMEK" not in s_upper:
             for a in adaylar:
                v = sayi_temizle(a)
                if v > 0 and v < 500000 and not (v < 50 and float(v).is_integer()): veriler['Kredi'] = max(veriler['Kredi'], v)
             if i+1 < len(satirlar):
                for a in re.findall(r'[\d\.,]+', satirlar[i+1]):
                    v = sayi_temizle(a)
                    if v > 0 and v < 500000 and not (v < 50 and float(v).is_integer()): veriler['Kredi'] = max(veriler['Kredi'], v)

        if "%" in s_upper or "TOPLAM" in s_upper or "KDV" in s_upper:
            v = 0.0
            for a in adaylar:
                val = sayi_temizle(a)
                if val > 0 and val < 500000: v = val
            
            if v > 0:
                if "KDV" in s_upper: veriler['KDV'] = max(veriler['KDV'], v)
                elif "TOPLAM" in s_upper or "MATRAH" in s_upper:
                    if "20" in s_upper: veriler['Matrah_20'] = max(veriler['Matrah_20'], v)
                    elif "10" in s_upper: veriler['Matrah_10'] = max(veriler['Matrah_10'], v)
                    elif " 1 " in s_upper: veriler['Matrah_1'] = max(veriler['Matrah_1'], v)
                    elif " 0 " in s_upper: veriler['Matrah_0'] = max(veriler['Matrah_0'], v)

    # Toplam Tutar Sağlaması
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    if hesaplanan > 0: veriler['Toplam'] = hesaplanan
    if veriler['KDV'] > veriler['Toplam']: veriler['KDV'] = 0.0
    
    return veriler

# --- 2. YÖNTEM: YOLO AI ANALİZİ ---
def yolo_analiz(results, img_np):
    veriler = {'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0}
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = detector.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop
            h, w, _ = img_np.shape
            y1, y2 = max(0, y1-5), min(h, y2+5)
            x1, x2 = max(0, x1-5), min(w, x2+5)
            cropped = img_np[y1:y2, x1:x2]
            
            # Oku
            ocr_res = reader.ocr(cropped, cls=False)
            text = " ".join([line[1][0] for line in ocr_res[0]]) if ocr_res and ocr_res[0] else ""
            
            if cls_name == 'tarih': veriler['Tarih'] = text
            elif cls_name == 'z_no': veriler['Z_No'] = re.sub(r'[^\d]', '', text)
            elif cls_name in ['toplam', 'nakit', 'kredi']:
                val = sayi_temizle(text)
                if cls_name == 'toplam': veriler['Toplam'] = max(veriler['Toplam'], val)
                elif cls_name == 'nakit': veriler['Nakit'] = max(veriler['Nakit'], val)
                elif cls_name == 'kredi': veriler['Kredi'] = max(veriler['Kredi'], val)
            elif 'kdv' in cls_name:
                val = sayi_temizle(text)
                if '10' in cls_name: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                elif '20' in cls_name: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)
                elif '1' in cls_name: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)

    if veriler['Nakit'] + veriler['Kredi'] > veriler['Toplam']:
        veriler['Toplam'] = veriler['Nakit'] + veriler['Kredi']
        
    return veriler

# --- ARAYÜZ VE AKIŞ ---
st.title("🪂 Z Raporu AI - V102 (Hibrid)")

uploaded_files = st.file_uploader("Fiş Yükle", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("Analiz Et"):
    tum_veriler = []
    for f in uploaded_files:
        img = Image.open(f)
        img_np = np.array(img)
        if len(img_np.shape) == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 1. ÖNCE YOLO'YU DENE
        yolo_results = detector(img_np, conf=0.20) # Eşik değerini düşürdük (Daha hassas)
        
        # YOLO bir şey buldu mu?
        if yolo_results and len(yolo_results[0].boxes) > 2: # En az 3 kutu bulmalı (Z No, Tarih, Tutar)
            veri = yolo_analiz(yolo_results, img_np)
            veri['Metod'] = "🤖 AI"
        else:
            # BULAMADIYSA KLASİK YÖNTEME GEÇ (YEDEK PARAŞÜT)
            ocr_res = reader.ocr(img_np, cls=False)
            satirlar = paddle_sonuclari_duzenle(ocr_res)
            veri = klasik_analiz(satirlar)
            veri['Metod'] = "🔎 Klasik"
        
        veri['Dosya'] = f.name
        if veri['Toplam'] > 0: veri['Durum'] = "✅"
        else: veri['Durum'] = "❌"
        tum_veriler.append(veri)
        
    df = pd.DataFrame(tum_veriler)
    cols = ["Durum", "Metod", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
    st.data_editor(df[[c for c in cols if c in df.columns]])
