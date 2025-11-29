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
st.set_page_config(page_title="Z Raporu AI (V112 - Final)", page_icon="✅", layout="wide")

# --- 1. MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    # YOLO Kontrolü (Varsa yükle, yoksa geç)
    detector = None
    if os.path.exists("best.pt"):
        detector = YOLO('best.pt')
    
    # PaddleOCR Yükle (Hata veren parametreler silindi)
    reader = PaddleOCR(use_angle_cls=True, lang='tr') 
    
    return detector, reader

try:
    detector, reader = load_models()
except Exception as e:
    st.error(f"Sistem Başlatma Hatası: {e}")
    st.stop()

# --- 2. GÖRÜNTÜ FORMATLAMA ---
def resmi_standartlastir(pil_image):
    # PIL -> Numpy
    image = np.array(pil_image)
    
    # Eğer Gri ise RGB yap
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Eğer PNG (4 kanal) ise RGB yap
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
    return image

# --- 3. SAYI TEMİZLEME ---
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

# --- 4. ANALİZ MOTORU ---
def verileri_isle(ocr_results, dosya_adi):
    veriler = {
        'Dosya': dosya_adi,
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    # Boş sonuç kontrolü
    if not ocr_results or ocr_results[0] is None:
        return veriler

    raw_data = ocr_results[0]
    
    # Veriyi temizle
    valid_data = []
    text_list = []
    for item in raw_data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = item[1][0]
            if text:
                valid_data.append(item)
                text_list.append(text)
    
    if not valid_data: return veriler

    full_text = " ".join(text_list).upper()
    
    # A. TARİH VE Z NO
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    zno = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno: veriler['Z_No'] = zno.group(1)

    # B. PARA ANALİZİ (KOORDİNATLI)
    # Yüksekliğe göre sırala
    valid_data = sorted(valid_data, key=lambda x: x[0][0][1])

    for i, item in enumerate(valid_data):
        bbox = item[0]
        text = item[1][0].upper()
        
        if "KUM" in text or "KÜM" in text or "YEKÜN" in text: continue

        def yanindaki_degeri_bul(index_no):
            try:
                mevcut_y = (valid_data[index_no][0][0][1] + valid_data[index_no][0][2][1]) / 2
                en_iyi_deger = 0.0
                for j in range(index_no + 1, len(valid_data)):
                    comp_box = valid_data[j][0]
                    comp_text = valid_data[j][1][0]
                    comp_y = (comp_box[0][1] + comp_box[2][1]) / 2
                    
                    if abs(mevcut_y - comp_y) < 20: # Tolerans
                        val = sayi_temizle(comp_text)
                        if val > 0 and val < 500000:
                            if not (val < 50 and float(val).is_integer()):
                                if val > en_iyi_deger: en_iyi_deger = val
                    else:
                        if (comp_y - mevcut_y) > 30: break
                return en_iyi_deger
            except:
                return 0.0

        if "NAKİT" in text or "NAKIT" in text:
            val = yanindaki_degeri_bul(i)
            if val > veriler['Nakit']: veriler['Nakit'] = val
            
        if ("KREDİ" in text or "KART" in text) and "YEMEK" not in text:
            val = yanindaki_degeri_bul(i)
            if val > veriler['Kredi']: veriler['Kredi'] = val

        if ("TOPLAM" in text or "GENEL" in text) and not any(x in text for x in ["KDV", "%", "VERGİ"]):
            val = yanindaki_degeri_bul(i)
            if val > veriler['Toplam']: veriler['Toplam'] = val

        if "%" in text or "TOPLAM" in text or "KDV" in text:
            val = yanindaki_degeri_bul(i)
            if val > 0:
                if "KDV" in text: veriler['KDV'] = max(veriler['KDV'], val)
                elif "TOPLAM" in text or "MATRAH" in text:
                    if "20" in text: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)
                    elif "10" in text: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                    elif " 1 " in text: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                    elif " 0 " in text: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)

    # FİNAL SAĞLAMA
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    if hesaplanan > veriler['Toplam']: veriler['Toplam'] = hesaplanan
    if veriler['KDV'] > veriler['Toplam']: veriler['KDV'] = 0.0

    return veriler

# --- ARAYÜZ ---
st.title("✅ Z Raporu AI - V112 (Hatasız)")

uploaded_files = st.file_uploader("Fiş Yükle", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("Analiz Et"):
    tum_veriler = []
    bar = st.progress(0)
    
    for i, f in enumerate(uploaded_files):
        try:
            img = Image.open(f)
            img_std = resmi_standartlastir(img)
            
            # --- BURASI DÜZELTİLDİ (PARAMETRESİZ) ---
            ocr_result = reader.ocr(img_std)
            # ----------------------------------------
            
            veri = verileri_isle(ocr_result, f.name)
            
            if veri['Toplam'] > 0: veri['Durum'] = "✅"
            else: veri['Durum'] = "❌"
            tum_veriler.append(veri)
            
        except Exception as e:
            st.warning(f"Hata ({f.name}): {e}")
        
        bar.progress((i+1)/len(uploaded_files))
        
    df = pd.DataFrame(tum_veriler)
    if not df.empty:
        cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
        # Sütunları kontrol et ve varsa göster
        mevcut_cols = [c for c in cols if c in df.columns]
        st.data_editor(df[mevcut_cols], num_rows="dynamic")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu.xlsx")
