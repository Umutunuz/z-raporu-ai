import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
import re
import io
import cv2
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V94 - Yapısal Denetçi)", page_icon="🏗️", layout="wide")

# --- PADDLE OCR MOTORU ---
@st.cache_resource
def load_paddle():
    return PaddleOCR(use_angle_cls=True, lang='tr')

try:
    reader = load_paddle()
except Exception as e:
    st.error(f"PaddleOCR Başlatılamadı: {e}")
    st.stop()

# --- GÖRÜNTÜ İŞLEME ---
def resmi_hazirla(pil_image):
    image = np.array(pil_image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
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

# --- SATIR BİRLEŞTİRİCİ (GÜÇLENDİRİLMİŞ) ---
def paddle_sonuclari_duzenle(results):
    # 1. Boş Veri Kontrolü
    if results is None or len(results) == 0 or results[0] is None:
        return []
    
    ocr_data = results[0]
    
    # 2. Yapısal Doğrulama (Bozuk verileri temizle)
    temiz_veri = []
    for item in ocr_data:
        # item yapısı: [ [[x,y],..], ("text", conf) ]
        # En az 2 elemanı olmalı ve koordinatları tam olmalı
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            box = item[0]
            text_info = item[1]
            if isinstance(box, (list, tuple)) and len(box) >= 1:
                temiz_veri.append(item)
    
    if not temiz_veri: return []

    # 3. Sıralama ve Birleştirme
    try:
        sorted_res = sorted(temiz_veri, key=lambda x: x[0][0][1])
    except:
        return [] # Sıralamada hata olursa boş dön

    satirlar = []
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

# --- MATEMATİK MOTORU ---
def en_iyi_kombinasyonu_bul(adaylar):
    nakit_list = sorted(list(set(adaylar.get('nakit', []) + [0.0])), reverse=True)
    kredi_list = sorted(list(set(adaylar.get('kredi', []) + [0.0])), reverse=True)
    toplam_list = sorted(list(set(adaylar.get('toplam', []))), reverse=True)
    
    en_iyi_set = {'Nakit': 0.0, 'Kredi': 0.0, 'Toplam': 0.0, 'Score': 0}

    for n in nakit_list:
        for k in kredi_list:
            hesaplanan = n + k
            for t in toplam_list:
                if t > 0 and abs(hesaplanan - t) < 1.5:
                    return {'Nakit': n, 'Kredi': k, 'Toplam': t, 'Score': 100}
    
    if en_iyi_set['Score'] == 0:
        max_n = nakit_list[0] if nakit_list else 0.0
        max_k = kredi_list[0] if kredi_list else 0.0
        
        valid_totals = [t for t in toplam_list if t < 500000]
        max_t = max(valid_totals) if valid_totals else (max_n + max_k)
        
        if abs((max_n + max_k) - max_t) < 2.0:
             return {'Nakit': max_n, 'Kredi': max_k, 'Toplam': max_t, 'Score': 90}
             
        return {'Nakit': max_n, 'Kredi': max_k, 'Toplam': max_t, 'Score': 50}

    return en_iyi_set

# --- ANALİZ MOTORU ---
def veri_analiz(satirlar):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    if not satirlar: return veriler

    full_text = " ".join(satirlar).upper()
    
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    zno = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR\s*NO|FİŞ\s*NO)\D{0,5}(\d+)', full_text)
    if zno: veriler['Z_No'] = zno.group(1)

    aday_havuzu = {'nakit': [], 'kredi': [], 'toplam': []}
    
    for i, s in enumerate(satirlar):
        s_upper = s.upper()
        if "KUM" in s_upper or "KÜM" in s_upper or "YEKÜN" in s_upper: continue
        
        adaylar = re.findall(r'[\d\.,]+', s_upper)
        
        if "NAKİT" in s_upper or "NAKIT" in s_upper:
            for aday in adaylar:
                val = sayi_temizle(aday)
                if val > 0 and not (val < 50 and float(val).is_integer()):
                    aday_havuzu['nakit'].append(val)
            if i+1 < len(satirlar):
                sub_adaylar = re.findall(r'[\d\.,]+', satirlar[i+1])
                for aday in sub_adaylar:
                    val = sayi_temizle(aday)
                    if val > 0 and not (val < 50 and float(val).is_integer()):
                        aday_havuzu['nakit'].append(val)

        if ("KREDİ" in s_upper or "KART" in s_upper) and "YEMEK" not in s_upper:
            for aday in adaylar:
                val = sayi_temizle(aday)
                if val > 0 and not (val < 50 and float(val).is_integer()):
                    aday_havuzu['kredi'].append(val)
            if i+1 < len(satirlar):
                sub_adaylar = re.findall(r'[\d\.,]+', satirlar[i+1])
                for aday in sub_adaylar:
                    val = sayi_temizle(aday)
                    if val > 0 and not (val < 50 and float(val).is_integer()):
                        aday_havuzu['kredi'].append(val)

        if ("TOPLAM" in s_upper or "GENEL" in s_upper) and not any(x in s_upper for x in ["KDV", "%", "VERGİ"]):
            for aday in adaylar:
                val = sayi_temizle(aday)
                if val > 0 and val < 500000:
                    aday_havuzu['toplam'].append(val)

        if "%" in s_upper or "TOPLAM" in s_upper or "KDV" in s_upper:
            val = 0.0
            for aday in adaylar:
                v = sayi_temizle(aday)
                if v > 0 and v < 500000: val = v
            
            if val > 0:
                if "KDV" in s_upper: veriler['KDV'] = max(veriler['KDV'], val)
                elif "TOPLAM" in s_upper or "MATRAH" in s_upper:
                    if "20" in s_upper: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)
                    elif "10" in s_upper: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                    elif " 1 " in s_upper: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                    elif " 0 " in s_upper: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)

    sonuc = en_iyi_kombinasyonu_bul(aday_havuzu)
    veriler['Nakit'] = sonuc['Nakit']
    veriler['Kredi'] = sonuc['Kredi']
    veriler['Toplam'] = sonuc['Toplam']
    
    if veriler['KDV'] > veriler['Toplam']: veriler['KDV'] = 0.0

    return veriler

# --- ARAYÜZ ---
st.title("🏗️ Z Raporu AI - V94 (Yapısal Denetçi)")

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
                
                result = reader.ocr(img_np)
                satirlar = paddle_sonuclari_duzenle(result)
                
                veri = veri_analiz(satirlar)
                veri['Dosya'] = name
                
                if veri['Toplam'] > 0: veri['Durum'] = "✅"
                else: veri['Durum'] = "❌"
                
                tum_veriler.append(veri)
            except Exception as e:
                # HATA DURUMUNDA BOŞ SATIR EKLE AMA ÇÖKME
                st.warning(f"Okuma Hatası: {name} - Lütfen fotoğrafı kontrol edin.")
            
            bar.progress((i+1)/len(resimler))
            
        df = pd.DataFrame(tum_veriler)
        if not df.empty:
            cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
            mevcut = [c for c in cols if c in df.columns]
            st.dataframe(df[mevcut], use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df[mevcut].to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
