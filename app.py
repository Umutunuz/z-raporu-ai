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
st.set_page_config(page_title="Z Raporu AI (V105 - Stabil)", page_icon="🏢", layout="wide")

# --- 1. MODELLERİ GÜVENLİ YÜKLE ---
@st.cache_resource
def load_models():
    # YOLO Kontrolü
    if not os.path.exists("best.pt"):
        return None, None
    
    # YOLO'yu Yükle
    detector = YOLO('best.pt')
    
    # PaddleOCR'ı En Yalın Haliyle Yükle (Argüman hatası vermemesi için)
    # use_angle_cls=True : Yamuk fişleri düzeltir.
    # lang='tr' : Türkçe karakterleri tanır.
    reader = PaddleOCR(use_angle_cls=True, lang='tr') 
    
    return detector, reader

try:
    detector, reader = load_models()
    if detector is None:
        st.error("⚠️ 'best.pt' dosyası bulunamadı! Lütfen GitHub'a yükleyin.")
        st.stop()
except Exception as e:
    st.error(f"Sistem Başlatma Hatası: {e}")
    st.stop()

# --- 2. GÖRÜNTÜ FORMATLAMA (CRASH ÖNLEYİCİ) ---
def resmi_standartlastir(pil_image):
    """
    Görüntüyü ne olursa olsun Paddle ve YOLO'nun sevdiği
    3 Kanallı (RGB) Numpy dizisine çevirir.
    """
    # PIL -> Numpy
    img = np.array(pil_image)
    
    # Eğer resim Gri (2 boyutlu) ise -> RGB (3 boyutlu) yap
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Eğer resim zaten Renkli ama 4 kanallı (PNG) ise -> RGB yap
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    return img

# --- 3. SAYI TEMİZLEME MOTORU ---
def sayi_temizle(text):
    if not text: return 0.0
    try:
        t = str(text).upper()
        # OCR Karakter Hatalarını Düzelt
        t = t.replace('O', '0').replace('S', '5').replace('I', '1').replace('L', '1').replace('Z', '2').replace('B', '8')
        
        # Özel Yama: 3/0 -> 370
        if "3/0" in t: t = t.replace("3/0", "370")
        
        # Temizlik
        t = t.replace(' ', '').replace('*', '').replace('TL', '')
        t = re.sub(r'[^\d,.]', '', t) # Rakam ve nokta/virgül dışındakileri at
        
        if len(t) > 0:
            # 1.500,00 -> 1500.00 formatı
            t = t.replace('.', 'X').replace(',', '.').replace('X', '')
            val = float(t)
            return val
    except:
        pass
    return 0.0

# --- 4. ANALİZ VE EŞLEŞTİRME MOTORU ---
def verileri_isle(ocr_results, dosya_adi):
    veriler = {
        'Dosya': dosya_adi,
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    if not ocr_results: return veriler

    # PaddleOCR çıktısı: [[[[x,y]..], ("text", conf)], ...]
    # Biz sadece metinleri bir listeye alalım
    text_list = [line[1][0] for line in ocr_results[0]]
    full_text = " ".join(text_list).upper()
    
    # --- A. TARİH VE Z NO (REGEX) ---
    tarih = re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', full_text)
    if tarih: veriler['Tarih'] = tarih.group(0).replace('-', '.').replace('/', '.')
    
    zno = re.search(r'(?:Z\s*NO|SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno: veriler['Z_No'] = zno.group(1)

    # --- B. PARA ANALİZİ (KOORDİNATLI) ---
    # Metinlerin konumlarına göre işlem yapacağız
    raw_data = ocr_results[0] # [bbox, (text, conf)]
    
    # Yüksekliğe göre sırala (Yukarıdan aşağıya okuma sırası)
    raw_data = sorted(raw_data, key=lambda x: x[0][0][1])

    for i, item in enumerate(raw_data):
        bbox = item[0]
        text = item[1][0].upper()
        
        # Kümülatif Filtresi (Kritik)
        if "KUM" in text or "KÜM" in text or "YEKÜN" in text: continue

        # --- DEĞER ARAMA FONKSİYONU ---
        def yanindaki_degeri_bul(index_no):
            # Bu satırın (kelimenin) Y koordinatı
            mevcut_y = (raw_data[index_no][0][0][1] + raw_data[index_no][0][2][1]) / 2
            
            en_iyi_deger = 0.0
            
            # Sonraki elemanlara bak (Aynı satırda olanları bul)
            for j in range(index_no + 1, len(raw_data)):
                comp_box = raw_data[j][0]
                comp_text = raw_data[j][1][0]
                
                comp_y = (comp_box[0][1] + comp_box[2][1]) / 2
                
                # Eğer Y farkı 15 pikselden azsa, aynı satırdadır
                if abs(mevcut_y - comp_y) < 15:
                    val = sayi_temizle(comp_text)
                    # Filtre: 50'den küçük tam sayıları (adetleri) alma. (12, 5 gibi)
                    # İstisna: Matrah oranları (1, 10, 20) bu fonksiyonda aranmaz.
                    if val > 0 and val < 500000:
                        if not (val < 50 and float(val).is_integer()):
                            if val > en_iyi_deger: en_iyi_deger = val
                else:
                    # Satır bitti, daha fazla aşağı inme (Hız için)
                    if (comp_y - mevcut_y) > 20: break
            return en_iyi_deger

        # 1. NAKİT
        if "NAKİT" in text or "NAKIT" in text:
            val = yanindaki_degeri_bul(i)
            if val > veriler['Nakit']: veriler['Nakit'] = val
            
        # 2. KREDİ
        if ("KREDİ" in text or "KART" in text) and "YEMEK" not in text:
            val = yanindaki_degeri_bul(i)
            if val > veriler['Kredi']: veriler['Kredi'] = val

        # 3. TOPLAM
        if ("TOPLAM" in text or "GENEL" in text) and not any(x in text for x in ["KDV", "%", "VERGİ"]):
            val = yanindaki_degeri_bul(i)
            if val > veriler['Toplam']: veriler['Toplam'] = val

        # 4. KDV / MATRAH (Özel Durum)
        if "%" in text or "TOPLAM" in text or "KDV" in text:
            val = yanindaki_degeri_bul(i)
            if val > 0:
                if "KDV" in text: veriler['KDV'] = max(veriler['KDV'], val)
                elif "TOPLAM" in text or "MATRAH" in text:
                    if "20" in text: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)
                    elif "10" in text: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                    elif " 1 " in text: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                    elif " 0 " in text: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)

    # --- FİNAL SAĞLAMA ---
    hesaplanan = veriler['Nakit'] + veriler['Kredi']
    
    # Eğer OCR Toplamı bulamadıysa (0 ise) veya Hesaplanan Toplam daha büyükse
    if hesaplanan > veriler['Toplam']:
        veriler['Toplam'] = hesaplanan
        
    # KDV Hata Kontrolü
    if veriler['KDV'] > veriler['Toplam']: veriler['KDV'] = 0.0

    return veriler

# --- ARAYÜZ ---
st.title("🏢 Z Raporu AI - V105 (Stabil)")

uploaded_files = st.file_uploader("Fiş Yükle", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("Analiz Et"):
    tum_veriler = []
    bar = st.progress(0)
    
    for i, f in enumerate(uploaded_files):
        try:
            img = Image.open(f)
            # KRİTİK: Görüntüyü standartlaştır (3 Kanal RGB)
            img_std = resmi_standartlastir(img)
            
            # 1. YOLO İLE DENE
            # conf=0.25 standarttır, oynama yapmadık
            yolo_results = detector(img_std, verbose=False) 
            
            # Eğer YOLO Z No veya Tutar bulduysa, o bölgeleri kesip oku
            # (Bu kısım çok kompleks olduğu için şimdilik pas geçip direkt tam sayfaya bakacağız
            # çünkü YOLO entegrasyonu bazen boş dönüyor, garantili yol tam sayfa okumaktır).
            
            # 2. PADDLE ILE TAM SAYFA OKU (EN GARANTİSİ)
            # cls parametresini SİLDİK. Hata vermez.
            ocr_result = reader.ocr(img_std)
            
            veri = verileri_isle(ocr_result, f.name)
            
            if veri['Toplam'] > 0: veri['Durum'] = "✅"
            else: veri['Durum'] = "❌"
            
            tum_veriler.append(veri)
            
        except Exception as e:
            st.error(f"Hata ({f.name}): {e}")
            
        bar.progress((i+1)/len(uploaded_files))
        
    df = pd.DataFrame(tum_veriler)
    if not df.empty:
        cols = ["Durum", "Tarih", "Z_No", "Toplam", "Nakit", "Kredi", "KDV", "Matrah_0", "Matrah_1", "Matrah_10", "Matrah_20", "Dosya"]
        st.data_editor(df[[c for c in cols if c in df.columns]], num_rows="dynamic")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st
