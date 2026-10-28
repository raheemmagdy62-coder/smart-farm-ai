"""Smart Farm AI - Phase 2: automatic plant identification (grape / guava / pear).

Model: BioCLIP 2 (zero-shot, loaded with open_clip).
It was NOT trained by us and has NOT been tested on our own leaf photos yet,
so every result is experimental.
"""

import hashlib
from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------
# Settings (easy to change later)
# ---------------------------------------------------------------
ALLOWED_TYPES = ["jpg", "jpeg", "png"]   # file extensions the uploader accepts
ALLOWED_FORMATS = {"JPEG", "PNG"}        # real formats we accept after opening
MAX_FILE_MB = 10                         # largest allowed image size
MIN_SIDE_PX = 100                        # smaller images get a warning

# To use the smaller / faster (but less accurate) model, use:
# MODEL_ID = "hf-hub:imageomics/bioclip"
MODEL_ID = "hf-hub:imageomics/bioclip-2"
INSTALL_COMMAND = "python -m pip install torch torchvision open_clip_torch"

# Decision thresholds. These are hand-picked starting values, NOT measured.
HIGH_CONFIDENCE = 0.80   # top score needed to call a result "likely"
LOW_CONFIDENCE = 0.50    # below this we say "cannot decide"
MIN_MARGIN = 0.25        # top score must beat the runner-up by this much

SUPPORTED = ["العنب", "الجوافة", "الكمثرى"]   # the only plants we claim to support
NOT_A_PLANT = "ليست ورقة نبات"
OTHER_PLANT = "نبات آخر"

# (Arabic label, scientific name). Several names can share one Arabic label;
# their scores are added together. The "other plants" are here so the model
# can say "this looks like something else" instead of being forced to pick
# one of our three plants.
SPECIES = [
    # --- supported plants ---
    ("العنب", "Vitis vinifera"),
    ("العنب", "Vitis labrusca"),
    ("الجوافة", "Psidium guajava"),
    ("الكمثرى", "Pyrus communis"),
    ("الكمثرى", "Pyrus pyrifolia"),
    ("الكمثرى", "Pyrus calleryana"),
    # --- other plants (not supported, used only as alternatives) ---
    ("التفاح", "Malus domestica"),
    ("السفرجل", "Cydonia oblonga"),
    ("الخوخ", "Prunus persica"),
    ("البرقوق", "Prunus domestica"),
    ("المشمش", "Prunus armeniaca"),
    ("التين", "Ficus carica"),
    ("المانجو", "Mangifera indica"),
    ("الزيتون", "Olea europaea"),
    ("الرمان", "Punica granatum"),
    ("البرتقال", "Citrus sinensis"),
    ("الأسكدنيا", "Eriobotrya japonica"),
    ("التوت", "Morus alba"),
    ("العناب", "Ziziphus jujuba"),
    ("الجمبولان", "Syzygium cumini"),
    (OTHER_PLANT, "Parthenocissus quinquefolia"),
    (OTHER_PLANT, "Hedera helix"),
    (OTHER_PLANT, "Solanum lycopersicum"),
    (OTHER_PLANT, "Zea mays"),
    (OTHER_PLANT, "Rosa chinensis"),
]

# Rough "this is not a plant" prompts. A species model may handle these
# poorly, so we never rely on them alone.
NON_PLANT_PROMPTS = [
    "a photo of a person.",
    "a photo of an animal.",
    "a photo of a building.",
    "a photo of a document or a screenshot.",
    "a photo of an object.",
]

st.set_page_config(page_title="Smart Farm AI", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------
# Design: right-to-left layout and green theme (CSS)
# ---------------------------------------------------------------
CSS = r"""<style>
.stApp {
    direction: rtl;
    background: linear-gradient(180deg, #f1f8ec 0%, #ffffff 55%);
}
.stApp, .stApp button, .stApp input {
    font-family: "Segoe UI", Tahoma, Arial, sans-serif;
}
.block-container { max-width: 1100px; padding-top: 2rem; }
footer { visibility: hidden; }

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    text-align: right;
    color: #1f3d2b;
}
.stApp h2, .stApp h3 { color: #1b5e20; }
[data-testid="stWidgetLabel"] p { color: #1f3d2b; font-weight: 600; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"], [data-testid="stExpander"] summary * { color: #2f4a39 !important; }
[data-testid="stCode"], pre, code { direction: ltr; text-align: left; }

/* Top banner */
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #43a047 60%, #81c784 100%);
    border-radius: 24px;
    padding: 2rem 2rem 1.6rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 24px rgba(27, 94, 32, 0.25);
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2.2rem;
    margin: 0 0 0.5rem 0;
    padding: 0;
    line-height: 1.5;
}
.hero p {
    color: #f1f8e9 !important;
    font-size: 1.05rem;
    line-height: 1.9;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.22);
    color: #ffffff;
    padding: 0.15rem 0.9rem;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-bottom: 0.7rem;
}

/* Three small step cards */
.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 0 0 1.5rem 0;
}
.step {
    background: #ffffff;
    border: 1px solid #d7ebd0;
    border-radius: 16px;
    padding: 1rem;
    text-align: center !important;
    box-shadow: 0 4px 12px rgba(27, 94, 32, 0.08);
}
.step .icon { font-size: 1.8rem; }
.step .title { color: #1b5e20; font-weight: 700; margin: 0.2rem 0; }
.step .text { color: #4b6b55; font-size: 0.9rem; }
@media (max-width: 700px) { .steps { grid-template-columns: 1fr; } }

/* Empty preview box */
.placeholder {
    border: 2px dashed #a5d6a7;
    border-radius: 16px;
    padding: 2.5rem 1rem;
    text-align: center;
    color: #4b6b55;
    background: #fbfef9;
}

/* Uploader: green look and Arabic wording (cosmetic only) */
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff;
    border: 2px dashed #66bb6a;
    border-radius: 16px;
}
[data-testid="stFileUploaderDropzoneInstructions"] * { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "اسحب الصورة وأفلتها هنا\A JPG أو JPEG أو PNG";
    white-space: pre-line;
    display: block;
    text-align: center;
    color: #2e5d3a;
}
[data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button * { display: none !important; }
[data-testid="stFileUploaderDropzone"] button::after {
    content: "اختيار صورة";
    font-size: 1rem;
}
</style>"""

HERO_HTML = """<div class="hero">
<span class="badge">المرحلة الثانية: التعرف على النبات فقط</span>
<h1>🌿 Smart Farm AI</h1>
<p>ارفع صورة لورقة نبات، وسيحاول النموذج التعرف عليها تلقائيًا (العنب أو الجوافة أو الكمثرى) مع عرض درجة الثقة. النتائج تجريبية، ولا يوجد كشف أمراض في هذه المرحلة.</p>
</div>"""

STEPS_HTML = """<div class="steps">
<div class="step"><div class="icon">📷</div><div class="title">ارفع الصورة</div><div class="text">اختر صورة واضحة لورقة واحدة</div></div>
<div class="step"><div class="icon">🤖</div><div class="title">تحليل تلقائي</div><div class="text">النموذج يقارن الورقة بعدة نباتات</div></div>
<div class="step"><div class="icon">📊</div><div class="title">النتيجة والثقة</div><div class="text">مع إظهار أعلى ثلاثة احتمالات</div></div>
</div>"""

PLACEHOLDER_HTML = """<div class="placeholder">🍃<br>ستظهر معاينة الصورة هنا بعد رفعها</div>"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)
st.markdown(STEPS_HTML, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Helper: safely open the uploaded file as an image (same as Phase 1)
# ---------------------------------------------------------------
def load_image(uploaded_file):
    """Return (image, error_message). Exactly one of them is None."""
    data = uploaded_file.getvalue()

    if len(data) == 0:
        return None, "الملف فارغ. يرجى اختيار صورة أخرى."
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, f"حجم الصورة كبير جدًا. الحد الأقصى {MAX_FILE_MB} ميجابايت."

    try:
        # First pass: check that the file really is a valid picture.
        with Image.open(BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()
        if image_format not in ALLOWED_FORMATS:
            return None, "نوع الملف غير مدعوم. يرجى رفع صورة JPG أو JPEG أو PNG."

        # Second pass: open it again for display and analysis.
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image)  # fix phone-photo rotation
        image = image.convert("RGB")
        return image, None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError):
        return None, "تعذّرت قراءة الصورة. قد يكون الملف تالفًا، يرجى تجربة صورة أخرى."


# ---------------------------------------------------------------
# The AI model
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load BioCLIP 2 once and prepare the text side for every candidate.

    The first call downloads the model (large, one time only).
    The imports are inside the function so the page still opens
    (with a clear message) if the libraries are not installed yet.
    """
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model.eval()

    candidates = []  # list of (arabic_label, [text prompts])
    for label, sci_name in SPECIES:
        prompts = [
            f"a photo of {sci_name}.",
            f"a photo of the leaf of {sci_name}.",
            sci_name,
        ]
        candidates.append((label, prompts))
    for prompt in NON_PLANT_PROMPTS:
        candidates.append((NOT_A_PLANT, [prompt]))

    labels = []
    text_vectors = []
    with torch.inference_mode():
        for label, prompts in candidates:
            features = model.encode_text(tokenizer(prompts))
            features = features / features.norm(dim=-1, keepdim=True)
            mean = features.mean(dim=0)
            text_vectors.append(mean / mean.norm())
            labels.append(label)
        text_matrix = torch.stack(text_vectors)

    return model, preprocess, text_matrix, labels


def classify(image, bundle):
    """Return [(arabic_label, score), ...] sorted from highest to lowest.

    Scores add up to 1 over ALL candidates (our 3 plants + other plants +
    non-plants). A score is the model's own preference, not a measured accuracy.
    """
    import torch

    model, preprocess, text_matrix, labels = bundle

    with torch.inference_mode():
        batch = preprocess(image).unsqueeze(0)
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * (features @ text_matrix.T)
        scores = logits.softmax(dim=-1)[0].tolist()

    totals = {}
    for label, score in zip(labels, scores):
        totals[label] = totals.get(label, 0.0) + score
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def get_ranked(image, file_hash):
    """Run the model once per uploaded file (the result is remembered)."""
    saved = st.session_state.get("analysis")
    if saved and saved["hash"] == file_hash:
        return saved["ranked"]

    with st.spinner(
        "جارٍ تحميل النموذج وتحليل الصورة... "
        "قد يستغرق أول تشغيل عدة دقائق بسبب تنزيل النموذج (راقب نافذة الطرفية)."
    ):
        bundle = load_model()
        ranked = classify(image, bundle)

    st.session_state["analysis"] = {"hash": file_hash, "ranked": ranked}
    return ranked


def show_result(ranked):
    """Turn the scores into an honest message for the user."""
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    percent = f"{top_score * 100:.0f}%"

    if top_score < LOW_CONFIDENCE:
        st.warning(
            "**لا يمكن تحديد النبات بثقة.** النموذج غير متأكد. "
            "جرّب صورة أوضح لورقة واحدة وبإضاءة جيدة."
        )
    elif top_label == NOT_A_PLANT:
        st.warning("**لا تبدو الصورة ورقة نبات.** يرجى رفع صورة ورقة واضحة.")
    elif top_label not in SUPPORTED:
        st.warning(
            f"**الصورة أقرب إلى: {top_label}.** هذا النبات غير مدعوم حاليًا، "
            "ولذلك لا نعتبرها عنبًا أو جوافة أو كمثرى."
        )
    elif top_score >= HIGH_CONFIDENCE and margin >= MIN_MARGIN:
        st.success(f"**النتيجة المحتملة: {top_label}**")
    else:
        st.warning(
            f"**نتيجة غير مؤكدة:** تبدو الورقة أقرب إلى {top_label}، "
            "لكن الثقة ليست عالية. لا تعتمد على هذه النتيجة."
        )

    st.metric("درجة ثقة النموذج في أعلى احتمال", percent)

    st.markdown("**أعلى ثلاثة احتمالات:**")
    for label, score in ranked[:3]:
        st.progress(min(max(score, 0.0), 1.0), text=f"{label}: {score * 100:.0f}%")

    st.caption(
        "درجة الثقة هي تفضيل النموذج بين الخيارات المعروفة له، وليست نسبة دقة مُقاسة. "
        "النموذج عام ولم يُدرَّب ولم يُختبر على صور أوراقنا بعد، لذلك النتيجة تجريبية."
    )


# ---------------------------------------------------------------
# Page layout: upload on the right, preview and results on the left
# ---------------------------------------------------------------
col_inputs, col_output = st.columns(2, gap="large")

image = None
error_message = None
file_hash = None

with col_inputs:
    st.subheader("ارفع صورة الورقة")
    uploaded_file = st.file_uploader(
        "صورة الورقة",
        type=ALLOWED_TYPES,
        label_visibility="collapsed",
    )
    st.caption(f"الصيغ المسموحة: JPG وJPEG وPNG، بحجم أقصى {MAX_FILE_MB} ميجابايت.")

    if uploaded_file is not None:
        image, error_message = load_image(uploaded_file)
        if error_message:
            st.error(error_message)
        else:
            file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

    with st.expander("نصائح لتصوير أفضل"):
        st.markdown(
            "- صوّر ورقة واحدة واضحة وقريبة.\n"
            "- استخدم ضوء النهار وتجنب الظلال القوية.\n"
            "- اجعل الورقة في مركز الصورة وبلا حركة.\n"
            "- يفضل أن تكون الخلفية بسيطة."
        )

with col_output:
    st.subheader("معاينة الصورة")
    if image is not None:
        preview = image.copy()
        preview.thumbnail((1200, 1200))  # keeps the page fast with big photos
        st.image(preview, caption=f"{image.width} × {image.height} بكسل")
        if min(image.size) < MIN_SIDE_PX:
            st.warning("الصورة صغيرة جدًا، وقد لا تكون النتيجة موثوقة. جرّب صورة أوضح.")
    else:
        st.markdown(PLACEHOLDER_HTML, unsafe_allow_html=True)

    st.subheader("لوحة النتائج")
    with st.container(border=True):
        if image is None:
            if error_message:
                st.error("لا يمكن التحليل لأن الصورة غير صالحة.")
            else:
                st.info("ارفع صورة ورقة ليبدأ التعرف التلقائي على النبات.")
        else:
            try:
                ranked = get_ranked(image, file_hash)
            except ImportError as exc:
                st.error(
                    "مكتبات النموذج غير مثبّتة بعد. شغّل هذا الأمر في PowerShell "
                    "ثم أعد تشغيل الموقع:"
                )
                st.code(INSTALL_COMMAND, language="powershell")
                with st.expander("تفاصيل تقنية"):
                    st.code(f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - show any problem kindly
                st.error(
                    "تعذّر تحميل النموذج أو تشغيله. تأكد من اتصال الإنترنت عند أول تشغيل "
                    "ومن وجود مساحة كافية على القرص، ثم جرّب مرة أخرى."
                )
                with st.expander("تفاصيل تقنية"):
                    st.code(f"{type(exc).__name__}: {exc}")
            else:
                show_result(ranked)

st.caption(
    "لا يرسل هذا التطبيق صورتك إلى أي خدمة خارجية ولا يحفظها في ملف. "
    "(يُنزَّل النموذج من الإنترنت مرة واحدة فقط.) "
    "المشروع تعليمي، ولن يغني مستقبلًا عن استشارة مهندس زراعي."
)