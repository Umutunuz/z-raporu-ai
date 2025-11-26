import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V79 - Keskin Nişancı)", page_icon="🎯", layout="wide")

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
    # Biraz daha keskinleştirme
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0) 
    return gray

# --- SAYI TEMİZLEME ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
        # Harf hataları
        t = t.replace('O', '0').replace('S', '5').replace('I', '1').replace('L', '1').replace('Z', '2').replace('B', '8')
        # 3/0 Yaması
        if "3/0" in t: t = t.replace("3/0", "370")
        
        t = t.replace(' ', '').replace('*', '').replace('TL', '')
        t = re.sub(r'[^\d,.]', '', t)
        
        if len(t) > 0:
            t = t.replace('.', 'X').replace(',', '.').replace('X', '')
            return float(t)
    except:
        pass
    return 0.0

# --- ANALİZ MOTORU ---
def veri_analiz(text_list):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    full_text = " ".join(text_list).upper()
    
    # 1. TARİH
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    # 2. Z NO
    zno_match = re.search(r'(?:Z\s*NO|Z\s*SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno_match: veriler['Z_No'] = zno_match.group(1)
    
    # YASAKLI KELİMELER LİSTESİ (Vergi No, Mersis No vb.)
    yasakli_kelimeler = ["VN", "VKN", "TCKN", "MERSIS", "SICIL", "TEL", "FAX", "KUM", "KÜM", "YEKÜN"]
    yasakli_indexler = []

    for i, text in enumerate(text_list):
        t = text.upper()
        if any(y in t for y in yasakli_kelimeler):
            yasakli_indexler.extend([i, i+1]) # Kendisi ve yanındakini yasakla

    # 3. DETAYLI ARAMA
    for i, text in enumerate(text_list):
        if i in yasakli_indexler: continue
        t = text.upper()
        
        # NAKİT
        if "NAKİT" in t or "NAKIT" in t:
            for j in range(1, 5):
                if i+j < len(text_list):
                    val = sayi_temizle(text_list[i+j])
                    if val > 0 and val < 500000:
                        if val < 50 and val.is_integer(): continue # Adet filtresi
                        veriler['Nakit'] = max(veriler['Nakit'], val)

        # KREDİ
        if ("KREDİ" in t or "KART" in t) and "YEMEK" not in t:
            for j in range(1, 5):
                if i+j < len(text_list):
                    val = sayi_temizle(text_list[i+j])
                    if val > 0 and val < 500000:
                        if val < 50 and val.is_integer(): continue
                        veriler['Kredi'] = max(veriler['Kredi'], val)

        # MATRAH / KDV
        if "%" in t or "TOPLAM" in t or "KDV" in t:
            val = 0.0
            for j in range(1, 4):
                if i+j < len(text_list):
                    v = sayi_temizle(text_list[i+j])
                    if v > 0 and v < 500000:
                        val = v
                        break
            
            if val > 0:
                if "KDV" in t: 
                    if val < 50000: veriler['KDV'] += val # Aşırı büyük KDV olmaz
                elif "TOPLAM" in t or "MATRAH" in t:
                    if "20" in t: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)
                    elif "10" in t: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                    elif " 1 " in t: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                    elif " 0 " in t: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)

    # 4. FİNAL HESAPLAMA (EN GÜVENLİ YÖNTEM)
    # Önce Nakit + Kredi toplamına bak.
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    
    if hesaplanan > 0:
        veriler['Toplam'] = hesaplanan
    else:
        # Eğer hesaplayamadıysak OCR'daki "TOPLAM" satırına bak (Yedek)
        # Ama asla rastgele en büyük sayıyı alma!
        ocr_toplam = 0.0
        for i, t in enumerate(text_list):
            if "TOPLAM" in t.upper() and not any(x in t.upper() for x in ["KDV", "%", "KUM"]):
                for j in range(1, 4):
                    if i+j < len(text_list):
                        v = sayi_temizle(text_list[i+j])
                        if v > ocr_toplam and v < 500000: ocr_toplam = v
        
        if ocr_toplam > 0: veriler['Toplam'] = ocr_toplam

    return veriler

# --- ARAYÜZ ---
st.title("🎯 Z Raporu AI - V79 (Keskin Nişancı)")

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
                img_np = resmi_hazirla(img)
                
                ocr_results = reader.readtext(img_np, detail=0)
                veri = veri_analiz(ocr_results)
                veri['Dosya'] = name
                
                # Durum İkonu
                if veri['Toplam'] > 0: 
                    veri['Durum'] = "✅"
                else: 
                    veri['Durum'] = "❌"
                
                tum_veriler.append(veri)
            except Exception as e:
                st.error(f"Hata: {name} - {e}")
            
            bar.progress((i+1)/len(resimler))
            
        df = pd.DataFrame(tum_veriler)
        if not df.empty:
            cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
            mevcut = [c for c in cols if c in df.columns]
            
            # EDİTÖR
            edited_df = st.data_editor(df[mevcut], num_rows="dynamic", use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
