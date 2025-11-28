import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import io
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V101 - Final)", page_icon="🤖", layout="wide")

# --- 1. MODELLERİ YÜKLE (ÖNBELLEK) ---
@st.cache_resource
def load_models():
    # A. Nesne Tanıma (Senin Eğittiğin Model)
    # best.pt dosyası GitHub'da app.py ile aynı klasörde olmalı
    if not os.path.exists("best.pt"):
        st.error("⚠️ 'best.pt' dosyası bulunamadı! Lütfen GitHub'a yüklediğinizden emin olun.")
        st.stop()
        
    detector = YOLO('best.pt')
    
    # B. Yazı Okuma (PaddleOCR)
    # show_log parametresi kaldırıldı (Hata kaynağıydı)
    reader = PaddleOCR(use_angle_cls=True, lang='tr')
    
    return detector, reader

try:
    detector, reader = load_models()
except Exception as e:
    st.error(f"Modeller Yüklenirken Hata Oluştu: {e}")
    st.stop()

# --- SAYI TEMİZLEME ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
        # OCR hatalarını düzelt
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

# --- ANALİZ MOTORU (YOLO + PADDLE) ---
def analyze_image(image, filename):
    img_np = np.array(image)
    
    # 1. YOLO İLE NESNELERİ BUL
    results = detector(img_np)
    
    veriler = {
        'Dosya': filename,
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    # Herhangi bir nesne bulundu mu?
    if not results or len(results[0].boxes) == 0:
        st.warning(f"⚠️ {filename} dosyasında Z Raporu alanları tespit edilemedi. Fotoğraf net mi?")
        return veriler, img_np

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Sınıf Adı
            cls_id = int(box.cls[0])
            cls_name = detector.names[cls_id]
            conf = float(box.conf[0])
            
            if conf < 0.4: continue # Düşük güvenli tahminleri atla

            # Koordinatlar
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Resmi Kes (Crop) ve Genişlet
            h, w, _ = img_np.shape
            y1 = max(0, y1 - 5)
            y2 = min(h, y2 + 5)
            x1 = max(0, x1 - 5)
            x2 = min(w, x2 + 5)
            
            cropped_img = img_np[y1:y2, x1:x2]
            
            # 2. PADDLEOCR İLE OKU
            # cls=True parametresi kaldırıldı (Gerekirse eklenebilir ama bazen hata yapar)
            ocr_result = reader.ocr(cropped_img, cls=False)
            
            text = ""
            if ocr_result and ocr_result[0]:
                text = " ".join([line[1][0] for line in ocr_result[0]])
            
            # 3. VERİYİ KAYDET
            if cls_name == 'tarih':
                veriler['Tarih'] = text
            elif cls_name == 'z_no':
                z_clean = re.sub(r'[^\d]', '', text)
                veriler['Z_No'] = z_clean
            elif cls_name in ['toplam', 'nakit', 'kredi']:
                val = sayi_temizle(text)
                if cls_name == 'toplam': veriler['Toplam'] = max(veriler['Toplam'], val)
                elif cls_name == 'nakit': veriler['Nakit'] = max(veriler['Nakit'], val)
                elif cls_name == 'kredi': veriler['Kredi'] = max(veriler['Kredi'], val)
            
            # KDV / Matrah
            elif 'kdv' in cls_name or 'matrah' in cls_name:
                val = sayi_temizle(text)
                if '10' in cls_name: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                elif '20' in cls_name: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)
                elif '1' in cls_name: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                elif '0' in cls_name: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)
            
            # Görselleştirme (Kutuları çiz)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4. SAĞLAMA (Eksik varsa tamamla)
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    
    if veriler['Toplam'] == 0 and hesaplanan > 0:
        veriler['Toplam'] = hesaplanan
    
    # Durum Kontrolü
    if veriler['Toplam'] > 0: veriler['Durum'] = "✅"
    else: veriler['Durum'] = "❌"

    return veriler, img_np

# --- ARAYÜZ ---
st.title("🤖 Z Raporu AI - V101 (Özel Eğitimli)")
st.info("Bu sürüm sizin eğittiğiniz Yapay Zeka modelini kullanır.")

# Sekmeler
tab1, tab2 = st.tabs(["📁 Dosya Yükle", "📷 Kamera"])
resimler = []

with tab1:
    uploaded_files = st.file_uploader("Galeriden Seç", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files: resimler.append((f, f.name))

with tab2:
    camera_pic = st.camera_input("Fotoğraf Çek")
    if camera_pic: resimler.append((camera_pic, "Kamera_Gorseli.jpg"))

if resimler:
    if st.button("Analizi Başlat", type="primary"):
        tum_veriler = []
        bar = st.progress(0)
        
        for i, (img_file, name) in enumerate(resimler):
            try:
                img = Image.open(img_file)
                
                veri, islenmis_resim = analyze_image(img, name)
                tum_veriler.append(veri)
                
            except Exception as e:
                st.error(f"Hata: {name} - {e}")
            
            bar.progress((i+1)/len(resimler))
            
        df = pd.DataFrame(tum_veriler)
        if not df.empty:
            cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
            mevcut_cols = [c for c in cols if c in df.columns]
            
            edited_df = st.data_editor(df[mevcut_cols], num_rows="dynamic", use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
