import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re
import emoji
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Pastikan NLTK siap ----
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

# ---- Fungsi Preprocessing ----
def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def map_emoji(text):
    # Konversi emoji ke huruf/angka
    emoji_map = {
        "🅰️": "A",     "🅱️": "B",     "🅾️": "O",
        "🆎": "AB",    "🆑": "CL",    "🆘": "SOS",
        "🆔": "ID",    "🆚": "VS",
        "🅿️": "P",     "🆒": "COOL",  "🆓": "FREE",
        "🆕": "NEW",   "🆖": "NG",    "🆙": "UP",
        "🆗": "OK",    "0️⃣": "0",     "1️⃣": "1",
        "2️⃣": "2",     "3️⃣": "3",     "4️⃣": "4",
        "5️⃣": "5",     "6️⃣": "6",     "7️⃣": "7",
        "8️⃣": "8",     "9️⃣": "9",     "🔟": "10",
        "🅐": "A",      "🅑" : "B",     "🅒": "C",
        "🅓": "D",      "🅔" : "E",     "🅕": "F",
        "🅖": "G",      "🅗" : "H",     "🅘": "I",
        "🅙": "J",      "🅚" : "K",     "🅛": "L",
        "🅜": "M",      "🅝" : "N",     "🅞": "O",
        "🅟": "P",      "🅠" : "Q",     "🅡": "R",
        "🅢": "S",      "🅣" : "T",     "🅤": "U",
        "🅥": "V",      "🅦" : "W",     "🅧": "X",
        "🅨": "Y",      "🅩" : "Z",
        "⓪": "0",      "①": "1",       "②": "2",
        "③": "3",      "④": "4",       "⑤": "5",
        "⑥": "6",      "⑦": "7",       "⑧": "8",
        "⑨": "9",      "⑩": "10",      "➀": "1",
        "➁": "2",      "➂": "3",      "➃": "4",
        "➄": "5",      "➅": "6",      "➆": "7",
        "➇": "8",      "➈": "9",      "➉": "10",
        "➊": "1",      "➋": "2",      "➌": "3",
        "➍": "4",      "➎": "5",      "➏": "6",
        "➐": "7",      "➑" : "8",     "➒":"9",
        "➓": "10",     "🅰": "A",      "🅱": "B",
        "🅲": "C",      "🅳": "D",      "🅴": "E",
        "🅵": "F",      "🅶": "G",      "🅷": "H",
        "🅸": "I",      "🅹": "J",      "🅺": "K",
        "🅻": "L",      "🅼": "M",      "🅽": "N",
        "🅾": "O",      "🅿": "P",      "🆀": "Q",
        "🆁": "R",      "🆂": "S",      "🆃": "T",
        "🆄": "U",      "🆅": "V",      "🆆": "W",
        "🆇": "X",      "🆈": "Y",      "🆉": "Z"
    }
    for emo, repl in emoji_map.items():
        text = text.replace(emo, repl)
    text = emoji.replace_emoji(text, replace="")
    return text

def case_folding(text):
    return text.lower()

def normalization(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'([A-Za-z])\1{1,}', r'\1', text)
    return text.strip()

# ---- Kamus slang ----
slang_dict = {
    "bg": "bang",              "yg": "yang",               "ngak": "tidak",            "ngga": "tidak",
    "mkasih": "terimakasih",   "mksh": "terimakasih",      "uda": "sudah",             "jg": "juga",
    "gk": "tidak",             "ajh": "saja",              "syg": "sayang",            "bilek": "be like",
    "gi mana": "bagaimana",    "kalo": "kalau",            "klo": "jika",              "keknya": "sepertinya",
    "gabisa": "tidak bisa",    "emg": "memang",            "pdhal": "padahal",         "bgt": "sangat",
    "bnyak": "banyak",         "sma": "sama",              "kasian": "kasihan",        "makasih": "terimakasih",
    "pake": "pakai",           "dri": "dari",              "dlu": "dulu",              "cwe": "wanita",
    "gimana": "bagaimana",     "jd": "jadi",               "d": "di",                  "orng": "orang",
    "bngt": "sangat",          "pkonya": "pokoknya",       "otw": "on the way",        "btw": "by the way",
    "jdi": "jadi",             "ilang":"hilang",           "ig": "instagram",          "plis": "please",
    "pls": "please",           "besti": "besty",           "dh": "sudah",              "dah": "sudah",
    "thnks": "thanks",         "brani": "berani",          "jt": "juta",               "kayak": "seperti",
    "gada": "tidak ada",       "caer": "cair",             "pgn": "ingin",             "jepe": "jp",
    "nggk": "tidak",           "pny" : "punya",            "sc" : "scatter",           "pdahal": "padahal",
    "bat" : "sangat",          "nemuin": "menemukan",      "pdhl": "padahal",          "km": "kamu",
    "ak" : "aku",              "gw" : "saya",              "gue" : "saya",             "aing": "saya",
    "gua": "saya",             "qu": "aku",                "qyu" : "aku",              "demen": "suka",
    "skrng" : "sekarang",      "skrg": "sekarang",         "maen": "main",             "tpi": "teapi",
    "tp": "tetapi",            "gcor": "gacor",            "rzeki": "rezeki",          "haru": "hari",
    "ni": "ini",               "tu": "itu",                "tuh": "itu",               "ga" : "tidak",
    "gak" :"tidak",            "ngapa": "mengapa",         "kenapa": "mengapa",        "lu" : "kamu",
    "kalau": "jika",           "link": "situs",            "ling": "situs",            "shacrcing": "searching",
    "seketer": "scatter",      "s tus": "situs",           "g cor": "gacor",           "gac r": "gacor",
    "sit s": "situs",          "gogle": "google",          "emng": "memang",           "smalam": "semalam",
    "banget":"sangat",         "cu": "lucu",               "scttr": "scatter",         "s1tus": "situs",
    "gugel": "google",         "mhjng": "mahjong",         "mahyong": "mahjong",       "msi": "masih",
    "mdal": "modal",           "wed" : "wd",               "menanf": "menang",         "ling": "link",
    "stus": "situs",           "s1tus": "situs",           "depoin": "mendepositkan",  "gede": "besar",
    "gedek": "besar",          "gampang": "mudah",         "sung": "langsung",         "hbis": "habis",
    "mak": "ibu",              "nga" : "tidak",            "tmn": "teman",             "temen": "teman",
    "lgi": "lagi",             "g" : "tidak",              "gmn": "bagaimana",         "nih" : "ini",
    "di kasih": "dikasih",     "sits": "situs",            "kayanya": "sepertinya",    "nyoba": "mencoba",
    "nyangka": "menyangka",    "sbnyak": "sebanyak",       "gatau": "tidak tahu",      "tau": "tahu",
    "jgn": "jangan",           "gapapa": "tidak apa-apa",  "kaga": "tidak",            "ngasih": "memberi",
    "doang": "saja",           "malah": "justru",          "cuma": "hanya",            "dikasi": "diberi",
    "kyanya": "sepertinya",    "gamau": "tidak mau",       "tak": "tidak",             "nangis": "menangis",
    "masi": "masih",           "ttp": "tetap",             "tetep": "tetap",           "sm": "sama",
    "biar": "supaya",          "nyakitin": "menyakiti",    "tida": "tidak",            "nyesal": "menyesal",
    "gitu": "begitu",          "gtu": "begitu",            "bilang": "berkata",         "nyesel": "menyesal",
    "beneran": "benar-benar",  "bnaran": "benar-benar",    "aja": "saja",              "sring": "sering",
    "abis": "habis",           "wd": "withdraw",           "jp": "jackpot",            "jepe": "jackpot",
    "jpe" : "jackpot",         "wede": "withdraw",         "wde": "withdraw",          "org": "orang",
    "orng": "orang",           "depo": "deposit",          "tapi": "tetapi",           "rungkad": "hancur",
    "rungkat": "hancur",       "mulu": "melulu",           "scater": "scatter",        "mj": "mahjong",
    "kk": "kak",               "boong":"bohong",           "rece": "receh",            "mhjng2": "mahjong2",       
    "mhjng1": "mahjong1",       "mj2": "mahjong2",         "mj1": "mahjong1"
}
def replace_slang(text):
    for key, val in slang_dict.items():
        text = re.sub(r'\b' + re.escape(key) + r'\b', val, text)
    return text

keyword_judol_list = [
   "jackpot", "jp", "jepe", "withdraw",
   "wd", "wede", "scatter", "cuan",
   "zeus", "mahjong", "mj", "mahjong1",
   "mahjong2", "modal", "cair",
   "deposit", "gacor", "maxwin",
   "spaceman", "slot", "pg", "situs"
]
def detect_keyword(text):
    for kw in keyword_judol_list:
        if kw in text:
            return 1
    return 0

def detect_nominal(text):
    text = text.lower()

    # 1. Abaikan huruf+angka (ojk606, abc123)
    if re.search(r"[a-zA-Z]+\d+", text):
        return 0

    # 2. Abaikan pola skor (0–9, spasi, 0–9)
    if re.search(r"\b[0-9]\s+[0-9]\b", text):
        return 0

    # 3. Nominal dengan satuan (k/rb/m/jt/t)
    if re.search(r"\b\d+(k|rb|m|jt|t)\b", text):
        return 1

    # 4. Nominal besar tanpa satuan (≥ 100)
    if re.search(r"\b\d{3,}\b", text):
        return 1

    # 5. Selain itu → bukan nominal
    return 0

def detect_brand(text):

    # gabungkan semua kata kunci ke dalam regex
    kw_pattern = "|".join(keyword_judol_list)

    # pola 1: huruf + angka
    pattern1 = r"\b[a-zA-Z]{3,}\d{2,}\b"

    # pola 2: angka + huruf
    pattern2 = r"\b\d{2,}[a-zA-Z]{3,}\b"

    # pola 3: angka di tengah
    pattern3 = r"\b[a-zA-Z]+\d+[a-zA-Z]+\b"

    # pola 4: nama domain (.com .net .vip .id .asia)
    pattern4 = r"\b[a-zA-Z0-9]+(\.com|\.net|\.vip|\.org|\.asia|\.id|\.xyz)\b"

    # pola 5: gabungan kata khas judol + opsional angka
    pattern5 = rf"\b(?:(?:[a-z0-9]+(?:{kw_pattern})[a-z0-9]*)|(?:[a-z0-9]*{kw_pattern}[a-z0-9]+))\b"

    # kalau ada salah satu cocok → deteksi brand
    if (
        re.search(pattern1, text)
        or re.search(pattern2, text)
        or re.search(pattern3, text)
        or re.search(pattern4, text)
        or re.search(pattern5, text)
    ):
        return 1
    return 0

def detect_google_combined(text):
    # --- 1. Google word ---
    google_found = bool(re.search(
    r"(google|digoogle|digugel|digogle|digoggle|gogle|goggle|gugel|gogel|googel)",
    text))

    # --- 2. Top ranking words ---
    top_found = bool(re.search(
        r"(paling atas|no ?1|nomor ?1|urutan pertama|urutan ?1|paling pertama)",
        text
    ))

    # --- 3. Kata perintah ---
    command_words = r"(cari|search|searching|ketik|cek|liat|lihat|buka|masuk|klik)"

    # --- 4. Keyword judol regex dari list ---
    pattern_kw = r"(" + "|".join(keyword_judol_list) + r")"

    # ======LOGIKA KOMBINASI=======

    # (A) Google + Top Ranking
    if google_found and top_found:
        return 1

    # (B) Google + Brand Judol (pakai fungsi detect_brand kamu)
    if google_found and detect_brand(text) == 1:
        return 1

    # (C) Google + kata perintah + kata judol
    if google_found and re.search(command_words, text) and re.search(pattern_kw, text):
        return 1

    # (D) Google + keyword judol
    if google_found and re.search(pattern_kw, text):
        return 1

    return 0

def tokenizing(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

def stemming(tokens):
    return stemmer.stem(" ".join(tokens)).split()

def preprocess_text(text):
    #sanpai slang
    step_normalizeunicode = normalize_unicode(text)
    step_emoji = map_emoji(step_normalizeunicode)
    step_casefolding = case_folding(step_emoji)
    step_slang = replace_slang(step_casefolding)

    #deteksi fitur otomatis dari teks yg sudahh dipreproses sampe slang
    feature_keyword = detect_keyword(step_slang)
    feature_nominal = detect_nominal(step_slang)
    feature_brand = detect_brand(step_slang)
    feature_google = detect_google_combined(step_slang)
    
    #lanjut preprocesing dari slang
    step_normalization = normalization(step_slang)
    step_token = tokenizing(step_normalization)
    step_stopwords = remove_stopwords(step_token)
    step_stemming = stemming(step_stopwords)
    clean_text = " ".join(step_stemming)

    return feature_keyword, feature_nominal, feature_brand, feature_google, clean_text

# ---- Load model & tokenizer ----
model = tf.keras.models.load_model("model3.keras")
with open("tokenizer3.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ==================== AESTHETIC PURPLE STYLE ====================
st.set_page_config(page_title="Deteksi Judol", layout="centered")

st.markdown("""
<style>

body {
    background: linear-gradient(145deg, #eee6ff, #f5f0ff, #f0e8ff);
}

.main-wrapper {
    max-width: 480px;
    margin: auto;
    padding: 10px;
}

/* Header Title */
.title-box {
    text-align: center;
    padding: 5px 20px;
    margin-bottom: 18px;
}

.title-text {
    font-size: 28px;
    font-weight: 800;
    color: #5b3fa3;
    letter-spacing: 0.5px;
}

.subtitle-text {
    font-size: 14px;
    margin-top: -4px;
    color: #7b62bd;
}

/* Aesthetic Card */
.card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 7px 26px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0px 8px 25px rgba(120, 85, 180, 0.13);
}

/* Textarea */
textarea {
    border-radius: 14px !important;
    border: 1px solid #d3c7f7 !important;
    background: #faf7ff !important;
}


/* Button purple aesthetic mirip default */
div.stButton > button {
    width: 120% !important;
    background: #f3eaff !important;              
    color: #5b3fa3 !important;                    
    border: 1px solid #c9b3f5 !important;         
    border-radius: 10px !important;               
    padding: 10px 0px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    box-shadow: 0px 4px 10px rgba(120, 85, 180, 0.15) !important;
    transition: 0.15s ease-in-out !important;
}

/* Hover */
div.stButton > button:hover {
    background: #e5d6ff !important;              
    border-color: #b39aef !important;
    box-shadow: 0px 6px 14px rgba(120, 85, 180, 0.25) !important;
}

/* When clicked (active) */
div.stButton > button:active {
    background: #d7c4ff !important;
    border-color: #a182e8 !important;
}
            
/* Result Box */
.result-card {
    background: rgba(255,255,255,0.85);
    padding: 5px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(90, 60, 150, 0.12);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.7);
}

.result-title {
    font-size: 18px;
    font-weight: 700;
    color: #4d347f;
    margin-bottom: 10px;
}

.badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 22px;
    font-size: 17px;
    font-weight: 700;
}

/* aman */
.badge-aman {
    background: #e9ffef;
    color: #2e8646;
    border: 1px solid #c0f5ce;
}

/* judol */
.badge-bahaya {
    background: #ffe8f0;
    color: #d63863;
    border: 1px solid #ffc7d9;
}

/* small description */
.desc {
    font-size: 13px;
    color: #6d5ca8;
    text-align: center;
    padding: 6px 12px;
    margin-top: -8px;
}

</style>
""", unsafe_allow_html=True)

# ==================== CONTENT ====================
st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='title-box'>
    <div class='title-text'>🔮 Deteksi Komentar Judol</div>
    <div class='subtitle-text'>Deteksi otomatis menggunakan pemrosesan bahasa & fitur heuristik</div>
</div>
""", unsafe_allow_html=True)

# Description
st.markdown("<div class='desc'>Masukkan komentar dari TikTok. Sistem akan mendeteksi indikasi promosi judi online berdasarkan kata kunci terkait judi online, nominal transaksi, nama brand judi online, dan konteks kalimat yang mengarahkan pengguna untuk mengunjungi situs tertentu melalui pencarian Google.</div>", unsafe_allow_html=True)

# Input area
st.markdown("<div class='card'>", unsafe_allow_html=True)
input_text = st.text_area("Tulis komentar di sini:", height=150, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# Button
st.markdown("<div class='detect-btn'>", unsafe_allow_html=True)
detect_clicked = st.button("Deteksi Sekarang")
st.markdown("</div>", unsafe_allow_html=True)


# ==================== PREDIKSI ====================
if detect_clicked and input_text.strip() != "":

    # ---- Ambil fitur otomatis dan teks hasil preprocessing ----
    f_keyword, f_nominal, f_brand, f_google, clean_text = preprocess_text(input_text)

    # ---- Sequence ----
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=15)

    # ---- Prediksi ----
    prob = model.predict([
        padded,
        np.array([[f_keyword]]),
        np.array([[f_nominal]]),
        np.array([[f_brand]]),
        np.array([[f_google]])
    ])[0][0]

    # ---- Label & Persentase ----
    is_judol = prob >= 0.5
    persen = prob * 100 if is_judol else (100 - prob * 100)
    persen = round(persen, 2)

    # ==================== OUTPUT UI ====================
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<div class='result-title'>Hasil Deteksi</div>", unsafe_allow_html=True)

    if is_judol:
        st.markdown(
            f"<div class='badge badge-bahaya'>Terindikasi PROMOSI JUDOL ⚠ — {persen:.1f}%</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='badge badge-aman'>Terindikasi NON-JUDOL ✔ — {persen:.1f}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)