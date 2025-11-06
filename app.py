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

def replace_slang(text):
    for key, val in slang_dict.items():
        text = re.sub(r'\b' + re.escape(key) + r'\b', val, text)
    return text

def normalization(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'([A-Za-z])\1{1,}', r'\1', text)
    return text.strip()

def tokenizing(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

def stemming(tokens):
    return stemmer.stem(" ".join(tokens)).split()

def preprocess_text(text):
    text = normalize_unicode(text)
    text = map_emoji(text)
    text = case_folding(text)
    text = replace_slang(text)
    text = normalization(text)
    tokens = tokenizing(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)

# ---- Load model & tokenizer ----
model = tf.keras.models.load_model("model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ---- UI Styling ----
st.set_page_config(page_title="Deteksi Komentar Judi Online", page_icon="🎰", layout="centered")

st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
        padding: 2rem;
        border-radius: 12px;
    }
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 0.75rem;
    }
    .result-box {
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .result-yes {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .result-no {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("Deteksi Komentar Promosi Judol")
st.caption("Deteksi apakah komentar di Tiktok mengandung promosi judi online.")

# ---- Input utama ----
with st.container():
    st.subheader("📝 Masukkan Komentar")
    user_comment = st.text_area(
        "Ketik atau tempel komentar di bawah ini:",
        placeholder="Contoh: WD cepet banget di situs ini 😎 gacor parah!",
        height=130
    )

# ---- Tambahan info ----
st.divider()
st.subheader("🔍 Informasi Tambahan")
st.info("Isi informasi tambahan berikut agar hasil deteksi lebih akurat.")

col1, col2 = st.columns(2)
with col1:
    keyword_judol = st.selectbox(
        "Apakah komentar mengandung kata kunci promosi (WD, JP, gacor, dll)?",
        ["Tidak", "Ya"]
    )
with col2:
    brand_judol = st.selectbox(
        "Apakah komentar menyebut nama brand judi online?",
        ["Tidak", "Ya"]
    )

st.divider()

# ---- Tombol Deteksi ----
if st.button("Deteksi Sekarang", use_container_width=True):
    if user_comment.strip() == "":
        st.warning("⚠️ Komentar tidak boleh kosong!")
    else:
        clean_text = preprocess_text(user_comment)
        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=15)

        keyword_feat = np.array([[1 if keyword_judol == "Ya" else 0]])
        brand_feat = np.array([[1 if brand_judol == "Ya" else 0]])

        pred = model.predict([padded, keyword_feat, brand_feat])[0][0]

        st.markdown("### Hasil Deteksi")

        if pred >= 0.5:
            st.markdown(
                f"<div class='result-box result-yes'>Komentar terdeteksi sebagai <b>Promosi Judi Online</b> "
                f"({pred*100:.2f}%)</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box result-no'>Komentar terdeteksi sebagai <b>Non-Judol</b> "
                f"({(1 - pred)*100:.2f}%)</div>",
                unsafe_allow_html=True
            )
