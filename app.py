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
def veri_analiz(satirlar):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    full_text = " ".join(satirlar).upper()
    
    # Tarih
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    # Z No
    for s in satirlar:
        s_upper = s.upper()
        if "Z NO" in s_upper or "Z SAYAÇ" in s_upper:
            match = re.search(r'(?:Z\s*NO|SAYAÇ)[\s.:]*(\d+)', s_upper)
            if match: veriler['Z_No'] = match.group(1)

    # Para Analizi
    for s in satirlar:
        s_upper = s.upper()
        if "KUM" in s_upper or "KÜM" in s_upper or "YEKÜN" in s_upper: continue

        adaylar = re.findall(r'[\d\.,]+', s_upper)
        satir_parasi = 0.0
        for aday in adaylar:
            val = sayi_temizle(aday)
            if val < 50 and float(val).is_integer() and "*" not in s_upper: continue
            if val > 0 and val < 500000:
                if val > satir_parasi: satir_parasi = val

        if satir_parasi == 0: continue

        if "NAKİT" in s_upper or "NAKIT" in s_upper:
            veriler['Nakit'] = max(veriler['Nakit'], satir_parasi)

        if ("KREDİ" in s_upper or "KART" in s_upper) and "YEMEK" not in s_upper:
            veriler['Kredi'] = max(veriler['Kredi'], satir_parasi)

        if ("TOPLAM" in s_upper or "GENEL" in s_upper) and not any(x in s_upper for x in ["KDV", "%", "VERGİ"]):
            veriler['Toplam'] = max(veriler['Toplam'], satir_parasi)

        if "%" in s_upper or "TOPLAM" in s_upper or "KDV" in s_upper:
            oran = -1
            if "20" in s_upper: oran = 20
            elif "10" in s_upper: oran = 10
            elif " 1 " in s_upper or "%1" in s_upper: oran = 1
            elif " 0 " in s_upper or "%0" in s_upper: oran = 0
            
            if oran != -1:
                if "KDV" in s_upper: veriler['KDV'] += satir_parasi
                elif "TOPLAM" in s_upper or "MATRAH" in s_upper:
                    if oran == 0: veriler['Matrah_0'] = max(veriler['Matrah_0'], satir_parasi)
                    elif oran == 1: veriler['Matrah_1'] = max(veriler['Matrah_1'], satir_parasi)
                    elif oran == 10: veriler['Matrah_10'] = max(veriler['Matrah_10'], satir_parasi)
                    elif oran == 20: veriler['Matrah_20'] = max(veriler['Matrah_20'], satir_parasi)

    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    if veriler['Toplam'] == 0 and hesaplanan > 0:
        veriler['Toplam'] = hesaplanan
    elif hesaplanan > veriler['Toplam']:
        veriler['Toplam'] = hesaplanan

    return veriler

# --- ARAYÜZ ---
st.title("⬛ Z Raporu AI - V81 (Kara Kutu)")

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
                
                ocr_results = reader.readtext(img_np, detail=1)
                
                # SATIRLARI BİRLEŞTİR
                satirlar = grupla_ve_satir_yap(ocr_results)
                
                # --- ⬛ KARA KUTU ALANI ⬛ ---
                with st.expander(f"🔍 Kara Kutu: {name} (AI Ne Gördü?)"):
                    st.code("\n".join(satirlar)) # Ham metni göster
                # -----------------------------
                
                veri = veri_analiz(satirlar)
                veri['Dosya'] = name
                
                if veri['Toplam'] > 0: veri['Durum'] = "✅"
                else: veri['Durum'] = "❌"
                
                tum_veriler.append(veri)
            except Exception as e:
                st.error(f"Hata: {name} - {e}")
            
            bar.progress((i+1)/len(resimler))
            
        df = pd.DataFrame(tum_veriler)
        if not df.empty:
            cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
            mevcut = [c for c in cols if c in df.columns]
            
            st.info("Veriler hatalıysa tabloya çift tıklayıp düzeltebilirsiniz.")
            edited_df = st.data_editor(df[mevcut], num_rows="dynamic", use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
