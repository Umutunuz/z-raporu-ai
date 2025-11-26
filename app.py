import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V80 - Satır Birleştirici)", page_icon="🧬", layout="wide")

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
    # Gürültü temizliği
    gray = cv2.medianBlur(gray, 3)
    return gray

# --- SAYI TEMİZLEME ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
        # Harf düzeltmeleri
        t = t.replace('O', '0').replace('S', '5').replace('I', '1').replace('L', '1').replace('Z', '2').replace('B', '8')
        if "3/0" in t: t = t.replace("3/0", "370")
        
        # KRİTİK: Boşlukları sil (2 . 144 -> 2.144 olsun diye)
        t = t.replace(' ', '').replace('*', '').replace('TL', '')
        t = re.sub(r'[^\d,.]', '', t)
        
        if len(t) > 0:
            # 1.500,00 -> 1500.00 formatı
            t = t.replace('.', 'X').replace(',', '.').replace('X', '')
            return float(t)
    except:
        pass
    return 0.0

# --- 🧬 SATIR BİRLEŞTİRİCİ (CORE TECHNOLOGY) 🧬 ---
def grupla_ve_satir_yap(ocr_results, y_tolerans=15):
    """
    EasyOCR'ın dağınık kutularını (Bounding Box) Y eksenine göre gruplar.
    Aynı hizadaki kelimeleri birleştirip cümle yapar.
    """
    # Y koordinatına (Top) göre sırala
    # bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] -> y1'e göre
    sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
    
    satirlar = []
    if not sorted_results: return satirlar

    mevcut_satir = [sorted_results[0]]
    # Mevcut satırın ortalama Y yüksekliği
    mevcut_y = (sorted_results[0][0][0][1] + sorted_results[0][0][2][1]) / 2

    for i in range(1, len(sorted_results)):
        box, text, conf = sorted_results[i]
        y_orta = (box[0][1] + box[2][1]) / 2

        # Eğer yükseklik farkı azsa (aynı satırdalarsa)
        if abs(y_orta - mevcut_y) < y_tolerans:
            mevcut_satir.append(sorted_results[i])
        else:
            # Satır bitti, kaydet
            # X koordinatına göre (soldan sağa) sırala
            mevcut_satir.sort(key=lambda x: x[0][0][0])
            # Metinleri birleştir
            satir_metni = " ".join([item[1] for item in mevcut_satir])
            satirlar.append(satir_metni)
            
            # Yeni satıra başla
            mevcut_satir = [sorted_results[i]]
            mevcut_y = y_orta
            
    # Son satırı ekle
    if mevcut_satir:
        mevcut_satir.sort(key=lambda x: x[0][0][0])
        satirlar.append(" ".join([item[1] for item in mevcut_satir]))
        
    return satirlar

# --- PARA ARAMA (SATIR BAZLI) ---
def satir_bazli_para_bul(satirlar, baslangic_index):
    limit = min(baslangic_index + 4, len(satirlar)) # Altındaki 3 satıra bak
    en_iyi_para = 0.0
    
    for i in range(baslangic_index, limit):
        s = satirlar[i]
        if "NO" in s or "ADET" in s: continue # Adet satırını atla (Opsiyonel)
        
        # Satırdaki tüm sayıları bul
        # 2.144,00 gibi sayıları yakalamak için regex
        adaylar = re.findall(r'[\d\.,\s]+', s) 
        
        for aday in adaylar:
            deger = sayi_temizle(aday)
            
            # Adet filtresi (50'den küçük tam sayılar - 5, 12, 37)
            if deger < 50 and float(deger).is_integer():
                # İstisna: Yanında * varsa al
                if "*" not in s: continue
            
            if deger > 0 and deger < 1000000:
                if deger > en_iyi_para: en_iyi_para = deger
                
    return en_iyi_para

# --- ANALİZ MOTORU ---
def veri_analiz(satirlar):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    # 1. TARİH VE Z NO
    full_text = " ".join(satirlar).upper()
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    zno = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno: veriler['Z_No'] = zno.group(1)

    # 2. DETAYLI ARAMA
    for i, s in enumerate(satirlar):
        s_upper = s.upper()
        
        # Kümülatif Engeli
        if "KUM" in s_upper or "KÜM" in s_upper or "YEKÜN" in s_upper: continue

        # NAKİT
        if "NAKİT" in s_upper or "NAKIT" in s_upper:
            tutar = satir_bazli_para_bul(satirlar, i)
            if tutar > veriler['Nakit']: veriler['Nakit'] = tutar

        # KREDİ
        if ("KREDİ" in s_upper or "KART" in s_upper) and "YEMEK" not in s_upper:
            tutar = satir_bazli_para_bul(satirlar, i)
            if tutar > veriler['Kredi']: veriler['Kredi'] = tutar

        # TOPLAM
        if ("TOPLAM" in s_upper or "GENEL" in s_upper) and not any(x in s_upper for x in ["KDV", "%", "VERGİ"]):
            tutar = satir_bazli_para_bul(satirlar, i)
            if tutar > veriler['Toplam']: veriler['Toplam'] = tutar

        # MATRAH / KDV
        if "%" in s_upper or "TOPLAM" in s_upper or "MATRAH" in s_upper or "KDV" in s_upper:
            tutar = satir_bazli_para_bul(satirlar, i)
            if tutar == 0: continue
            
            oran = -1
            if "20" in s_upper: oran = 20
            elif "10" in s_upper: oran = 10
            elif " 1 " in s_upper or "%1" in s_upper: oran = 1
            elif " 0 " in s_upper or "%0" in s_upper: oran = 0
            
            if oran != -1:
                if "KDV" in s_upper: veriler['KDV'] += tutar
                elif "TOPLAM" in s_upper or "MATRAH" in s_upper:
                    # Matrah toplamdan büyük olamaz
                    if veriler['Toplam'] == 0 or tutar < veriler['Toplam']:
                        if oran == 0: veriler['Matrah_0'] = max(veriler['Matrah_0'], tutar)
                        elif oran == 1: veriler['Matrah_1'] = max(veriler['Matrah_1'], tutar)
                        elif oran == 10: veriler['Matrah_10'] = max(veriler['Matrah_10'], tutar)
                        elif oran == 20: veriler['Matrah_20'] = max(veriler['Matrah_20'], tutar)

    # 3. SAĞLAMA
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    if veriler['Toplam'] == 0 and hesaplanan > 0:
        veriler['Toplam'] = hesaplanan
    elif hesaplanan > veriler['Toplam']:
        veriler['Toplam'] = hesaplanan

    return veriler

# --- ARAYÜZ ---
st.title("🧬 Z Raporu AI - V80 (Satır Birleştirici)")

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
                
                # AI OKUMASI (Koordinatlı - detail=1)
                ocr_results = reader.readtext(img_np, detail=1)
                
                # SİHİR BURADA: Dağınık parçaları SATIR haline getir
                satirlar = grupla_ve_satir_yap(ocr_results)
                
                # HATA AYIKLAMA: Ne okuduğunu görmek istersen aç
                with st.expander(f"🔍 Kara Kutu: {name}"):
                    st.code("\n".join(satirlar))
                
                veri = veri_analiz(satirlar)
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
            
            st.info("Tablodaki verilere çift tıklayıp düzeltebilirsiniz.")
            edited_df = st.data_editor(df[mevcut], num_rows="dynamic", use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False)
            st.download_button("📥 Excel İndir", buffer.getvalue(), "Z_Raporu_AI.xlsx")
