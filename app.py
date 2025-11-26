import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V80 - Lazer)", page_icon="📏", layout="wide")

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
    # Gürültü temizliği ve keskinleştirme
    gray = cv2.medianBlur(gray, 3)
    return gray

# --- SAYI TEMİZLEME ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
        t = t.replace('O', '0').replace('S', '5').replace('I', '1').replace('L', '1').replace('Z', '2').replace('B', '8')
        
        # 3/0 Yaması (Senin bozuk fiş için)
        if "3/0" in t: t = t.replace("3/0", "370")
        
        t = t.replace(' ', '').replace('*', '').replace('TL', '')
        t = re.sub(r'[^\d,.]', '', t)
        
        if len(t) > 0:
            t = t.replace('.', 'X').replace(',', '.').replace('X', '')
            return float(t)
    except:
        pass
    return 0.0

# --- LAZER HİZALAMA FONKSİYONU (Pixel-Perfect) ---
def ayni_satirdaki_sayiyi_bul(hedef_kelime_box, tum_sonuclar):
    """
    Hedef kelimenin tam sağında ve AYNI HİZADA olan sayıyı bulur.
    """
    # Hedefin Y koordinatının ortası
    # box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    hedef_y_orta = (hedef_kelime_box[0][1] + hedef_kelime_box[2][1]) / 2
    hedef_x_sag = hedef_kelime_box[1][0] # Kelimenin bittiği yer (Sağ taraf)

    en_yakin_sayi = 0.0
    en_yakin_mesafe = 10000 # X ekseninde ne kadar yakın?

    for bbox, text, conf in tum_sonuclar:
        # Kendi kendisiyle karşılaştırma
        if bbox == hedef_kelime_box: continue

        # Sayı olup olmadığına bak
        val = sayi_temizle(text)
        
        # --- ADET FİLTRESİ ---
        # Eğer sayı 50'den küçükse ve tam sayıysa (12, 5 gibi) ALMA.
        # Ancak yanında * varsa al.
        if val < 50 and float(val).is_integer() and "*" not in text:
            continue
            
        # Koordinat Kontrolü
        sayi_y_orta = (bbox[0][1] + bbox[2][1]) / 2
        sayi_x_sol = bbox[0][0]

        # 1. YÜKSEKLİK KONTROLÜ (Aynı satırda mı?)
        # Tolerans: 15 piksel (Satır kaymalarına karşı hassas ama esnek)
        if abs(hedef_y_orta - sayi_y_orta) < 15:
            
            # 2. KONUM KONTROLÜ (Sağında mı?)
            if sayi_x_sol > hedef_x_sag:
                
                # 3. YAKINLIK KONTROLÜ (En yakınındaki sayıyı al)
                # Aynı satırda birden fazla sayı olabilir (Örn: %18 ... 500)
                mesafe = sayi_x_sol - hedef_x_sag
                if mesafe < en_yakin_mesafe:
                    en_yakin_mesafe = mesafe
                    en_yakin_sayi = val

    return en_yakin_sayi

# --- ANALİZ MOTORU ---
def veri_analiz(ocr_results):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    # Düz Metin (Tarih ve Z No için)
    text_list = [item[1] for item in ocr_results]
    full_text = " ".join(text_list).upper()
    
    # 1. TARİH
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    # 2. Z NO
    zno_match = re.search(r'(?:Z\s*NO|Z\s*SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno_match: veriler['Z_No'] = zno_match.group(1)

    # 3. KOORDİNATLI ARAMA
    for bbox, text, conf in ocr_results:
        t = text.upper()
        
        # Kümülatif Engeli
        if "KUM" in t or "KÜM" in t or "YEKÜN" in t: continue

        # NAKİT
        if "NAKİT" in t or "NAKIT" in t:
            # Sadece yanındakine bak (Lazer Hizalama)
            val = ayni_satirdaki_sayiyi_bul(bbox, ocr_results)
            if val > veriler['Nakit']: veriler['Nakit'] = val

        # KREDİ
        if ("KREDİ" in t or "KART" in t) and "YEMEK" not in t:
            val = ayni_satirdaki_sayiyi_bul(bbox, ocr_results)
            if val > veriler['Kredi']: veriler['Kredi'] = val

        # MATRAH / KDV
        if "%" in t or "TOPLAM" in t or "MATRAH" in t or "KDV" in t:
            val = ayni_satirdaki_sayiyi_bul(bbox, ocr_results)
            
            if val > 0:
                if "KDV" in t: 
                    veriler['KDV'] += val
                elif "TOPLAM" in t or "MATRAH" in t:
                    # Oranı bul (Satırın içinde %1, %10 var mı?)
                    oran = -1
                    if "20" in t: oran = 20
                    elif "10" in t: oran = 10
                    elif " 1 " in t or "%1" in t: oran = 1
                    elif " 0 " in t or "%0" in t: oran = 0
                    
                    if oran == 0: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)
                    elif oran == 1: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                    elif oran == 10: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                    elif oran == 20: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)

    # 4. TOPLAM TUTAR VE SAĞLAMA
    # En güvenli yöntem: Parçaları topla.
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    
    if hesaplanan > 0:
        veriler['Toplam'] = hesaplanan
    else:
        # Eğer parçalar 0 ise (okunamadıysa), "TOPLAM" yazısının yanına bak
        for bbox, text, conf in ocr_results:
            t = text.upper()
            if ("TOPLAM" in t or "GENEL" in t) and not any(x in t for x in ["KDV", "%", "KUM"]):
                val = ayni_satirdaki_sayiyi_bul(bbox, ocr_results)
                if val > veriler['Toplam'] and val < 500000:
                    veriler['Toplam'] = val

    return veriler

# --- ARAYÜZ ---
st.title("📏 Z Raporu AI - V80 (Lazer Hizalama)")

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
                
                # AI OKUMASI (Detail=1 Koordinatları verir)
                ocr_results = reader.readtext(img_np, detail=1)
                
                veri = veri_analiz(ocr_results)
                veri['Dosya'] = name
                
                # Durum
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
            
            edited_df = st.data_editor(df[mevcut], num_rows="dynamic", use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
