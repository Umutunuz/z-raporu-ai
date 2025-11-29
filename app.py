import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pytesseract
import re
import io
import cv2
import os
import shutil

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V110 - Tesseract)", page_icon="🦅", layout="wide")

# --- TESSERACT AYARLARI ---
@st.cache_resource
def get_tesseract_cmd():
    # Linux (Sunucu) için yol
    path = shutil.which("tesseract")
    if path: return path
    return "tesseract"

pytesseract.pytesseract.tesseract_cmd = get_tesseract_cmd()

# --- GÖRÜNTÜ İŞLEME (TESSERACT İÇİN ÖZEL) ---
def resmi_hazirla(pil_image):
    image = np.array(pil_image)
    # Griye çevir
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Büyütme (Tesseract küçük yazıları sevmez)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. Gürültü Temizliği
    gray = cv2.medianBlur(gray, 3)
    
    # 3. Threshold (Keskin Siyah-Beyaz)
    # Bu işlem silik yazıları koyulaştırır
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(thresh)

# --- SAYI TEMİZLEME ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
        t = t.replace('O', '0').replace('S', '5').replace('I', '1').replace('L', '1').replace('Z', '2').replace('B', '8')
        # Bozuk fiş yaması
        if "3/0" in t: t = t.replace("3/0", "370")
        
        t = t.replace(' ', '').replace('*', '').replace('TL', '')
        t = re.sub(r'[^\d,.]', '', t)
        
        if len(t) > 0:
            t = t.replace('.', 'X').replace(',', '.').replace('X', '')
            return float(t)
    except:
        pass
    return 0.0

# --- VERİ AYIKLAMA (TESSERACT ÇIKTISINDAN) ---
def veri_analiz(raw_text):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    # OCR çıktısındaki yaygın kelime hatalarını düzelt
    full_text = raw_text.upper()
    full_text = full_text.replace("LGPLAM", "TOPLAM").replace("LGLKOÜY", "TOPKDV")
    
    satirlar = full_text.split('\n')
    
    # 1. TARİH
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    # 2. Z NO
    zno = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno: veriler['Z_No'] = zno.group(1)

    # 3. SATIR SATIR ANALİZ
    for i, s in enumerate(satirlar):
        s = s.strip()
        if not s: continue
        if "KUM" in s or "KÜM" in s or "YEKÜN" in s: continue

        # O satırdaki paraları bulma fonksiyonu
        def satirdaki_paralar(satir_metni):
            adaylar = re.findall(r'[\d\.,]+', satir_metni)
            paralar = []
            for a in adaylar:
                val = sayi_temizle(a)
                if val > 0 and val < 500000:
                    # 50'den küçük tam sayıları (adetleri) ele
                    if val < 50 and float(val).is_integer() and "*" not in satir_metni: continue
                    paralar.append(val)
            return paralar

        paralar = satirdaki_paralar(s)
        if not paralar: continue
        max_para = max(paralar)

        # NAKİT
        if "NAKİT" in s or "NAKIT" in s:
            veriler['Nakit'] = max(veriler['Nakit'], max_para)
            # Alt satıra da bak (Tesseract bazen parayı alta atar)
            if i+1 < len(satirlar):
                alt_paralar = satirdaki_paralar(satirlar[i+1])
                if alt_paralar: veriler['Nakit'] = max(veriler['Nakit'], max(alt_paralar))

        # KREDİ
        if ("KREDİ" in s or "KART" in s) and "YEMEK" not in s:
            veriler['Kredi'] = max(veriler['Kredi'], max_para)
            if i+1 < len(satirlar):
                alt_paralar = satirdaki_paralar(satirlar[i+1])
                if alt_paralar: veriler['Kredi'] = max(veriler['Kredi'], max(alt_paralar))

        # TOPLAM
        if ("TOPLAM" in s or "GENEL" in s) and not any(x in s for x in ["KDV", "%", "VERGİ"]):
            veriler['Toplam'] = max(veriler['Toplam'], max_para)

        # KDV / MATRAH
        if "%" in s or "TOPLAM" in s or "KDV" in s:
            if "KDV" in s: 
                veriler['KDV'] = max(veriler['KDV'], max_para)
            elif "TOPLAM" in s or "MATRAH" in s:
                if "20" in s: veriler['Matrah_20'] = max(veriler['Matrah_20'], max_para)
                elif "10" in s: veriler['Matrah_10'] = max(veriler['Matrah_10'], max_para)
                elif " 1 " in s: veriler['Matrah_1'] = max(veriler['Matrah_1'], max_para)
                elif " 0 " in s: veriler['Matrah_0'] = max(veriler['Matrah_0'], max_para)

    # 4. SAĞLAMA
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    if hesaplanan > veriler['Toplam']: veriler['Toplam'] = hesaplanan
    if veriler['KDV'] > veriler['Toplam']: veriler['KDV'] = 0.0

    return veriler, full_text

# --- ARAYÜZ ---
st.title("🦅 Z Raporu AI - V110 (Tesseract)")

uploaded_files = st.file_uploader("Fiş Yükle", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("Analiz Et"):
    tum_veriler = []
    bar = st.progress(0)
    
    for i, f in enumerate(uploaded_files):
        try:
            img = Image.open(f)
            # Görüntüyü işle
            img_processed = resmi_hazirla(img)
            
            # Tesseract ile Oku (PSM 6: Blok Metin Modu)
            custom_config = r'--oem 3 --psm 6'
            raw_text = pytesseract.image_to_string(img_processed, lang='tur', config=custom_config)
            
            # Analiz Et
            veri, ham_metin = veri_analiz(raw_text)
            veri['Dosya'] = f.name
            
            if veri['Toplam'] > 0: veri['Durum'] = "✅"
            else: veri['Durum'] = "❌"
            
            tum_veriler.append(veri)
            
            # Hata ayıklama için metni göster (İstersen)
            # with st.expander(f"🔍 Ne Okundu? - {f.name}"):
            #    st.text(ham_metin)
            
        except Exception as e:
            st.warning(f"Hata ({f.name}): {e}")
        
        bar.progress((i+1)/len(uploaded_files))
        
    df = pd.DataFrame(tum_veriler)
    if not df.empty:
        cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
        st.data_editor(df[[c for c in cols if c in df.columns]], num_rows="dynamic")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu.xlsx")
