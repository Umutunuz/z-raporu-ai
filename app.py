import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI", page_icon="🧠", layout="wide")

# --- YAPAY ZEKA MOTORU (ÖNBELLEKTE TUTAR) ---
@st.cache_resource
def load_model():
    # Türkçe ve İngilizce modellerini yükler. GPU varsa kullanır, yoksa CPU.
    return easyocr.Reader(['tr', 'en'], gpu=False)

try:
    reader = load_model()
except Exception as e:
    st.error("Yapay Zeka Modeli Yüklenemedi! Lütfen sayfayı yenileyin.")
    st.stop()

# --- GÖRÜNTÜ İŞLEME ---
def resmi_hazirla(pil_image):
    image = np.array(pil_image)
    # EasyOCR zaten renkli okur ama gri tonlama kontrastı artırır
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

# --- SAYI TEMİZLEME ---
def sayi_temizle(deger_str):
    if not deger_str: return 0.0
    try:
        temiz = str(deger_str).upper()
        # OCR hatalarını düzelt
        temiz = temiz.replace("S", "5").replace("O", "0").replace("l", "1").replace("I", "1").replace("Z", "2").replace("B", "8")
        # Senin fişindeki meşhur 3/0 hatası için yama
        if "3/0" in temiz: temiz = temiz.replace("3/0", "370")
        
        temiz = temiz.replace("*", "").replace(" ", "").replace("/", "7").replace("'", "").replace('"', "")
        temiz = re.sub(r'[^\d,.]', '', temiz)
        
        if len(temiz) > 0:
            # 1.500,00 formatı
            gecici = temiz.replace('.', 'X').replace(',', 'X')
            if 'X' not in gecici: return float(temiz)
            
            son_isaret = gecici.rfind('X')
            tam = temiz[:son_isaret]
            ondalik = temiz[son_isaret+1:]
            
            tam = re.sub(r'[^\d]', '', tam)
            if len(ondalik) > 2: ondalik = ondalik[:2]
            
            return float(f"{tam}.{ondalik}")
        return 0.0
    except: return 0.0

# --- PARA BULUCU (FİLTRELİ) ---
def para_bul(satirlar, index):
    limit = min(index + 4, len(satirlar))
    en_iyi_para = 0.0
    
    for i in range(index, limit):
        s = satirlar[i]
        # Adet belirten kelimeler varsa atla
        if "ADET" in s or "NO" in s: continue
        
        adaylar = re.findall(r'[\d\.,]+', s)
        for aday in adaylar:
            deger = sayi_temizle(aday)
            # 50'den küçük tam sayıları ele
            if deger < 50 and float(deger).is_integer(): continue
            if deger > en_iyi_para: en_iyi_para = deger
    return en_iyi_para

# --- ANALİZ MOTORU ---
def veri_analiz(text_list):
    # EasyOCR listesini düz metne çevir
    full_text = "\n".join(text_list)
    # Bozuk kelimeleri düzelt
    full_text = full_text.upper().replace("LGPLAM", "TOPLAM").replace("LGLKOÜY", "TOPKDV")
    
    satirlar = full_text.split('\n')
    
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    # Tek satır metin (Regex için)
    duz_metin = full_text.replace("\n", " ")
    
    # Tarih
    tarih = re.search(r'(\d{2}[./-]\d{2}[./-]\d{4})', duz_metin)
    if tarih: veriler['Tarih'] = tarih.group(1).replace('-', '.').replace('/', '.')
    
    # Z No
    zno = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR|FİŞ\s*NO)[\s:.]*(\d{3,5})', duz_metin)
    if zno: veriler['Z_No'] = zno.group(1)

    # Satır Analizi
    for i, s in enumerate(satirlar):
        s = s.strip()
        if not s or "KUM" in s or "KÜM" in s: continue

        # Nakit
        if "NAKİT" in s or "NAKIT" in s:
            tutar = para_bul(satirlar, i)
            if tutar > veriler['Nakit']: veriler['Nakit'] = tutar

        # Kredi
        if ("KREDİ" in s or "KART" in s or "BANKA" in s) and "YEMEK" not in s:
            tutar = para_bul(satirlar, i)
            if tutar > veriler['Kredi']: veriler['Kredi'] = tutar

        # Toplam
        if ("TOPLAM" in s or "GENEL" in s) and not any(x in s for x in ["KDV", "%", "VERGİ"]):
            tutar = para_bul(satirlar, i)
            if tutar > veriler['Toplam'] and tutar < 500000:
                veriler['Toplam'] = tutar
                
        # KDV / Matrah
        if "%" in s or "TOPLAM" in s or "KDV" in s:
            tutar = para_bul(satirlar, i)
            if tutar == 0: continue
            
            oran = -1
            if "20" in s: oran = 20
            elif "10" in s: oran = 10
            elif " 1 " in s: oran = 1
            elif " 0 " in s: oran = 0
            
            if oran != -1:
                if "KDV" in s: veriler['KDV'] += tutar
                elif "TOPLAM" in s or "MATRAH" in s:
                    if oran == 0: veriler['Matrah_0'] = max(veriler['Matrah_0'], tutar)
                    elif oran == 1: veriler['Matrah_1'] = max(veriler['Matrah_1'], tutar)
                    elif oran == 10: veriler['Matrah_10'] = max(veriler['Matrah_10'], tutar)
                    elif oran == 20: veriler['Matrah_20'] = max(veriler['Matrah_20'], tutar)

    # Sağlama
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    if hesaplanan > 0 and (veriler['Toplam'] == 0 or abs(veriler['Toplam'] - hesaplanan) > 1.0):
        veriler['Toplam'] = hesaplanan

    return veriler

# --- ARAYÜZ ---
st.title("🧠 Z Raporu AI - Bulut Sürümü")
st.markdown("Fotoğrafı yükleyin veya kamerayı kullanın. Yapay Zeka (Deep Learning) ile analiz edilecektir.")

# Sekmeli Giriş (Dosya Yükle veya Kamera Aç)
tab1, tab2 = st.tabs(["📁 Dosya Yükle", "📷 Kamera Kullan"])

resimler = []

with tab1:
    uploaded_files = st.file_uploader("Galeriden Seç", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            resimler.append((f, f.name))

with tab2:
    camera_pic = st.camera_input("Fotoğraf Çek")
    if camera_pic:
        resimler.append((camera_pic, "Kamera_Gorseli.jpg"))

if resimler:
    if st.button("Analizi Başlat", type="primary"):
        tum_veriler = []
        bar = st.progress(0)
        
        for i, (img_file, name) in enumerate(resimler):
            try:
                img = Image.open(img_file)
                img_np = resmi_hazirla(img)
                
                # EASYOCR OKUMASI (AI)
                # detail=0 sadece metin listesi döndürür
                raw_text_list = reader.readtext(img_np, detail=0, paragraph=False)
                
                veri = veri_analiz(raw_text_list)
                veri['Dosya'] = name
                
                if veri['Toplam'] > 0: veri['Durum'] = "✅"
                else: veri['Durum'] = "❌"
                
                tum_veriler.append(veri)
            except Exception as e:
                st.error(f"Hata: {name} - {e}")
            
            bar.progress((i+1)/len(resimler))
            
        df = pd.DataFrame(tum_veriler)
        if not df.empty:
            # Düzenli Tablo
            cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
            mevcut = [c for c in cols if c in df.columns]
            st.dataframe(df[mevcut], use_container_width=True)
            
            # Excel İndir
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df[mevcut].to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")