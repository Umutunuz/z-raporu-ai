import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V76 - Güvenli)", page_icon="🛡️", layout="wide")

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

# --- KOORDİNAT ANALİZİ ---
def satir_hizasinda_sayi_bul(anahtar_kelime_box, tum_sonuclar, tolerans=20):
    hedef_y = (anahtar_kelime_box[0][1] + anahtar_kelime_box[2][1]) / 2
    en_iyi_sayi = 0.0
    
    for bbox, text, conf in tum_sonuclar:
        deger = sayi_temizle(text)
        if deger <= 0: continue 
        
        sayi_y = (bbox[0][1] + bbox[2][1]) / 2
        
        # Y Koordinatı tutuyorsa (Aynı satırdaysa)
        if abs(hedef_y - sayi_y) < tolerans:
            # X Koordinatı büyükse (Yazının sağındaysa)
            if bbox[0][0] > anahtar_kelime_box[0][0]:
                
                # Adet ve Küçük Sayı Filtresi
                # Sadece 50'den küçük ve TAM sayı ise (12, 5 gibi) alma.
                # Ama 10.50 gibi bir sayıysa al.
                if deger < 50 and float(deger).is_integer(): 
                    # İstisna: Eğer yanında * varsa kesin al
                    if "*" not in text: continue

                if deger > en_iyi_sayi:
                    en_iyi_sayi = deger
                    
    return en_iyi_sayi

# --- Z NO BULUCU ---
def z_no_bul(full_text):
    match = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR\s*NO|FİŞ\s*NO)[\s:.]*(\d+)', full_text)
    if match: return match.group(1)
    
    match_eku = re.search(r'EKU\s*NO[\s:.]*(\d+)', full_text)
    if match_eku: return match_eku.group(1)
    return ""

# --- ANALİZ MOTORU ---
def veri_analiz(ocr_results):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    text_list = [item[1] for item in ocr_results]
    full_text = " ".join(text_list).upper()
    
    # 1. TARİH
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    # 2. Z NO
    veriler['Z_No'] = z_no_bul(full_text)

    # 3. PARALARI BUL (Koordinat Sistemi)
    for bbox, text, conf in ocr_results:
        t = text.upper()
        
        # --- KRİTİK KÜMÜLATİF FİLTRESİ ---
        # Eğer satırda KUM, KÜM, YEKÜN, TOPLAM SATIŞ (Genelde kümülatif başlığıdır) varsa o satıra bakma!
        if "KUM" in t or "KÜM" in t or "YEKÜN" in t or "TOPLAM SATIŞ" in t: 
            continue

        # NAKİT
        if "NAKİT" in t or "NAKIT" in t:
            bulunan = satir_hizasinda_sayi_bul(bbox, ocr_results)
            if bulunan > 0: veriler['Nakit'] = max(veriler['Nakit'], bulunan)

        # KREDİ
        if ("KREDİ" in t or "KART" in t) and "YEMEK" not in t:
            bulunan = satir_hizasinda_sayi_bul(bbox, ocr_results)
            if bulunan > 0: veriler['Kredi'] = max(veriler['Kredi'], bulunan)

        # GENEL TOPLAM
        # "KDV" ve "VERGİ" kelimesi olmayan "TOPLAM" satırına bak.
        if ("TOPLAM" in t or "GENEL" in t) and not any(x in t for x in ["KDV", "%", "VERGİ"]):
            bulunan = satir_hizasinda_sayi_bul(bbox, ocr_results)
            # Ekstra Güvenlik: 1 Milyon TL üstü günlük ciro olmaz (Kümülatiftir), alma.
            if bulunan > 0 and bulunan < 1000000: 
                veriler['Toplam'] = max(veriler['Toplam'], bulunan)

        # MATRAH / KDV
        if "%" in t or "TOPLAM" in t or "MATRAH" in t or "KDV" in t:
            bulunan = satir_hizasinda_sayi_bul(bbox, ocr_results)
            if bulunan > 0:
                if "KDV" in t: veriler['KDV'] += bulunan
                elif "20" in t: veriler['Matrah_20'] = max(veriler['Matrah_20'], bulunan)
                elif "10" in t: veriler['Matrah_10'] = max(veriler['Matrah_10'], bulunan)
                elif " 1 " in t: veriler['Matrah_1'] = max(veriler['Matrah_1'], bulunan)
                elif " 0 " in t: veriler['Matrah_0'] = max(veriler['Matrah_0'], bulunan)

    # 4. FİNAL DOĞRULAMA VE TAMAMLAMA (Matematiksel Güvenlik)
    # Asla rastgele en büyük sayıyı alma. Sadece parçaları topla.
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    
    # Eğer OCR Toplamı bulamadıysa (0 ise) -> Toplam = Nakit + Kredi
    if veriler['Toplam'] == 0 and hesaplanan > 0:
        veriler['Toplam'] = hesaplanan
        
    # Eğer OCR Toplamı buldu ama Nakit+Kredi toplamı ondan daha büyükse (Daha güvenilirse)
    elif hesaplanan > veriler['Toplam']:
        veriler['Toplam'] = hesaplanan

    return veriler

# --- ARAYÜZ ---
st.title("🛡️ Z Raporu AI - V76 (Güvenli)")

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
                veri = veri_analiz(ocr_results)
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
            st.dataframe(df[mevcut], use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df[mevcut].to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
