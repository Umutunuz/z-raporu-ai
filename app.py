import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V78 - Akıl Süzgeci)", page_icon="🧠", layout="wide")

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
    # Kontrastı artır
    gray = cv2.equalizeHist(gray)
    return gray

# --- SAYI TEMİZLEME ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
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

# --- AKIL SÜZGECİ (HATA ENGELLEYİCİ) ---
def mantik_kontrolu(veriler):
    """
    Mantıksız sayıları (Kümülatif vb.) temizler.
    """
    # 1. KDV KONTROLÜ: KDV, Toplamdan büyük olamaz!
    if veriler['KDV'] > veriler['Toplam']:
        veriler['KDV'] = 0.0 # Kümülatif çekmiş, sil.

    # 2. MATRAH KONTROLÜ: Matrah, Toplamdan büyük olamaz.
    for m in ['Matrah_0', 'Matrah_1', 'Matrah_10', 'Matrah_20']:
        if veriler[m] > veriler['Toplam']:
            veriler[m] = 0.0

    # 3. TOPLAM KONTROLÜ: Eğer Toplam 0 ise, Parçaları Topla
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    if veriler['Toplam'] == 0 and hesaplanan > 0:
        veriler['Toplam'] = hesaplanan
        
    return veriler

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
    if zno_match:
        veriler['Z_No'] = zno_match.group(1)
    
    # 3. SATIR BAZLI ARAMA
    
    # Önce tüm mantıklı sayıları bul (Genel Toplam Adayları)
    tum_sayilar = []
    for t in text_list:
        val = sayi_temizle(t)
        # 500.000'den küçük (Kümülatif olmayan) sayıları al
        if val > 0 and val < 500000: tum_sayilar.append(val)
        
    if tum_sayilar:
        veriler['Toplam'] = max(tum_sayilar)

    # Detaylı Arama
    for i, text in enumerate(text_list):
        t = text.upper()
        
        # Kümülatif Engeli
        if "KUM" in t or "KÜM" in t or "YEKÜN" in t: continue

        # NAKİT
        if "NAKİT" in t or "NAKIT" in t:
            for j in range(1, 5):
                if i+j < len(text_list):
                    val = sayi_temizle(text_list[i+j])
                    if val > 0 and val < 500000:
                        # Adet filtresi
                        if val < 50 and float(val).is_integer(): continue
                        if val <= veriler['Toplam']: veriler['Nakit'] = max(veriler['Nakit'], val)

        # KREDİ
        if ("KREDİ" in t or "KART" in t) and "YEMEK" not in t:
            for j in range(1, 5):
                if i+j < len(text_list):
                    val = sayi_temizle(text_list[i+j])
                    if val > 0 and val < 500000:
                        if val < 50 and float(val).is_integer(): continue
                        if val <= veriler['Toplam']: veriler['Kredi'] = max(veriler['Kredi'], val)

        # KDV / MATRAH
        if "%" in t or "TOPLAM" in t or "KDV" in t:
            # Yanındaki sayıyı bul
            val = 0.0
            for j in range(1, 4):
                if i+j < len(text_list):
                    v = sayi_temizle(text_list[i+j])
                    if v > 0 and v < 500000:
                        val = v
                        break
            
            if val > 0:
                if "KDV" in t: 
                    if val < veriler['Toplam']: veriler['KDV'] += val
                elif "TOPLAM" in t or "MATRAH" in t:
                    if "20" in t: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)
                    elif "10" in t: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                    elif " 1 " in t: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                    elif " 0 " in t: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)

    # --- FİNAL KONTROL ---
    veriler = mantik_kontrolu(veriler)
    
    return veriler

# --- ARAYÜZ ---
st.title("🧠 Z Raporu AI - V78 (Kontrollü)")

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
            
            # EDİTÖRÜ AKTİF ET (Kullanıcı elle düzeltebilsin)
            st.info("Tablo üzerindeki verilere çift tıklayarak düzeltebilirsiniz.")
            edited_df = st.data_editor(df[mevcut], num_rows="dynamic", use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
