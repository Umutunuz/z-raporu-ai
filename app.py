import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import easyocr
import re
import io
import cv2

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Z Raporu AI (V77 - Geometrik)", page_icon="📐", layout="wide")

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
        # Harf Düzeltmeleri
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

# --- KOORDİNAT EŞLEŞTİRİCİ (EN ÖNEMLİ KISIM) ---
def deger_bul_koordinatli(hedef_kelimeler, tum_veriler, yasakli_kelimeler=[]):
    """
    Hedef kelimeyi bulur (Örn: NAKİT).
    Onunla AYNI YÜKSEKLİKTE (Y Ekseni) ve SAĞINDA (X Ekseni) olan sayıyı alır.
    """
    bulunan_deger = 0.0
    en_iyi_y_farki = 1000 # En yakın satırı bulmak için
    
    # 1. Hedef Kelimenin Konumunu Bul
    hedef_box = None
    for bbox, text, conf in tum_veriler:
        t_upper = text.upper()
        if any(k in t_upper for k in hedef_kelimeler) and not any(y in t_upper for y in yasakli_kelimeler):
            hedef_box = bbox
            break # İlk bulduğunu al (Genelde en üstteki doğrudur)
            
    if not hedef_box: return 0.0

    # Hedefin Y (Dikey) Merkezi
    hedef_y = (hedef_box[0][1] + hedef_box[2][1]) / 2
    hedef_x = hedef_box[2][0] # Hedefin sağ ucu

    # 2. Aynı Hizadaki Sayıyı Ara
    for bbox, text, conf in tum_veriler:
        # Kendisi değilse
        if bbox == hedef_box: continue
        
        # Sayı mı?
        val = sayi_temizle(text)
        if val <= 0: continue
        
        # Adet Filtresi (50'den küçük tam sayıları alma - Matrah oranları hariç)
        if val < 50 and float(val).is_integer() and "MATRAH" not in str(hedef_kelimeler): 
            continue

        # Konum Kontrolü
        sayi_y = (bbox[0][1] + bbox[2][1]) / 2
        sayi_x = bbox[0][0] # Sayının sol ucu
        
        # Aynı satırda mı? (Y farkı az olmalı)
        y_farki = abs(hedef_y - sayi_y)
        
        # Sayı, yazının sağında mı?
        if y_farki < 30 and sayi_x > hedef_x: # 30 piksel tolerans
            # En yakın hizadakini seç
            if y_farki < en_iyi_y_farki:
                en_iyi_y_farki = y_farki
                bulunan_deger = val

    return bulunan_deger

# --- ANALİZ MOTORU ---
def veri_analiz(ocr_results):
    veriler = {
        'Tarih': "", 'Z_No': "", 'Toplam': 0.0, 'Nakit': 0.0, 'Kredi': 0.0, 
        'KDV': 0.0, 'Matrah_0': 0.0, 'Matrah_1': 0.0, 'Matrah_10': 0.0, 'Matrah_20': 0.0
    }
    
    # Düz Metin Listesi (Tarih ve Z No için)
    text_list = [item[1] for item in ocr_results]
    full_text = " ".join(text_list).upper()
    
    # 1. TARİH (Gelişmiş Regex - Boşlukları Yutar)
    # Örn: 16 . 10 . 2025 veya 16/10/2025
    tarih = re.search(r'(\d{2})\s*[./-]\s*(\d{2})\s*[./-]\s*(\d{4})', full_text)
    if tarih: 
        veriler['Tarih'] = f"{tarih.group(1)}.{tarih.group(2)}.{tarih.group(3)}"
    
    # 2. Z NO (Sadece "Z NO" kelimesinin yanındakini alır)
    # EKÜ, FİŞ NO gibi tuzaklara düşmez.
    zno_match = re.search(r'(?:Z\s*NO|Z\s*SAYAÇ|RAPOR\s*NO)\D{0,5}(\d+)', full_text)
    if zno_match:
        candidate = zno_match.group(1)
        # 37 gibi küçük sayıları Z No sanmasın (Genelde Fiş Nosudur)
        if int(candidate) > 0:
            veriler['Z_No'] = candidate

    # 3. TOPLAM TUTAR (İki Yöntem)
    # Yöntem A: "TOPLAM" yazısının sağındaki sayı
    tutar_geo = deger_bul_koordinatli(["TOPLAM", "GENEL"], ocr_results, ["KDV", "%", "VERGİ", "FİŞ", "KUM", "KÜM"])
    
    # Yöntem B: Sayfadaki en büyük sayı (Kümülatif hariç)
    max_val = 0.0
    for item in ocr_results:
        t = item[1].upper()
        if "KUM" in t or "KÜM" in t: continue
        v = sayi_temizle(t)
        if v > max_val and v < 500000: max_val = v
    
    veriler['Toplam'] = max(tutar_geo, max_val)

    # 4. NAKİT VE KREDİ (Geometrik Arama)
    veriler['Nakit'] = deger_bul_koordinatli(["NAKİT", "NAKIT"], ocr_results)
    veriler['Kredi'] = deger_bul_koordinatli(["KREDİ", "KART", "BANKA"], ocr_results, ["YEMEK"])

    # 5. MATRAH VE KDV (Oran + Geometri)
    # % işaretini bulup yanındakini alacağız
    for bbox, text, conf in ocr_results:
        t = text.upper()
        if "%" in t or "TOPLAM" in t or "KDV" in t:
            # Oran tespiti
            oran = -1
            if "20" in t: oran = 20
            elif "10" in t: oran = 10
            elif " 1 " in t or "%1" in t: oran = 1
            elif " 0 " in t or "%0" in t: oran = 0
            
            # Eğer oran bulduysak, o satırdaki parayı bul
            # Bu sefer kendi kutusunu hedef gösteriyoruz
            val = deger_bul_koordinatli([t], ocr_results) # Kendi satırındaki diğer sayıyı bul
            
            if val > 0 and val < veriler['Toplam']: # Matrah toplamdan büyük olamaz
                if "KDV" in t: veriler['KDV'] += val
                elif "TOPLAM" in t or "MATRAH" in t:
                    if oran == 0: veriler['Matrah_0'] = max(veriler['Matrah_0'], val)
                    elif oran == 1: veriler['Matrah_1'] = max(veriler['Matrah_1'], val)
                    elif oran == 10: veriler['Matrah_10'] = max(veriler['Matrah_10'], val)
                    elif oran == 20: veriler['Matrah_20'] = max(veriler['Matrah_20'], val)

    # 6. MUHASEBE KONTROLÜ (Kümülatif Temizliği)
    # Eğer KDV > Toplam ise, o KDV yanlıştır (Kümülatiftir), sıfırla.
    if veriler['KDV'] > veriler['Toplam']: veriler['KDV'] = 0.0

    # Eğer Nakit + Kredi > 0 ise ve Toplam'dan farklıysa, Toplamı güncelle
    toplam_odeme = veriler['Nakit'] + veriler['Kredi']
    if toplam_odeme >
