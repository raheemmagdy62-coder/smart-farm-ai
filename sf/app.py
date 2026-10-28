"""Smart Farm AI - version 5 (Arabic-first Streamlit app).

Run:  python -m streamlit run app.py
Tabs: plant identification, symptom check, Smart Crop Advisor, science and limits
All logic lives in plant_model.py / disease.py / diagnosis.py / crop_advisor.py / weather.py.
"""

import hashlib
from datetime import date
from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

import crop_advisor as ca
import diagnosis as dg
import disease
import plant_model as pm
import weather as wx

ALLOWED_TYPES = ["jpg", "jpeg", "png"]
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_FILE_MB = 10
INSTALL_COMMAND = "python -m pip install torch torchvision open_clip_torch"

GOVERNORATES = [
    "القاهرة", "الجيزة", "الإسكندرية", "القليوبية", "الشرقية", "الدقهلية", "البحيرة",
    "المنوفية", "الغربية", "كفر الشيخ", "دمياط", "بورسعيد", "الإسماعيلية", "السويس",
    "الفيوم", "بني سويف", "المنيا", "أسيوط", "سوهاج", "قنا", "الأقصر", "أسوان",
    "البحر الأحمر", "الوادي الجديد", "مطروح", "شمال سيناء", "جنوب سيناء",
]
OTHER_PLACE = "مكان آخر (اكتب الاسم)"

st.set_page_config(page_title="Smart Farm AI", page_icon="🌿", layout="wide")

CSS = r"""<style>
.stApp { direction: rtl; background: linear-gradient(180deg, #f1f8ec 0%, #ffffff 60%); }
.stApp, .stApp button, .stApp input, .stApp textarea { font-family: "Segoe UI", Tahoma, Arial, sans-serif; }
.block-container { max-width: 1150px; padding-top: 1.5rem; }
footer { visibility: hidden; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { text-align: right; color: #1f3d2b; }
.stApp h2, .stApp h3 { color: #1b5e20; }
[data-testid="stWidgetLabel"] p { color: #1f3d2b; font-weight: 600; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"], [data-testid="stExpander"] summary * { color: #2f4a39 !important; }
[data-testid="stCode"], pre, code { direction: ltr; text-align: left; }
.hero { background: linear-gradient(135deg, #1b5e20 0%, #43a047 60%, #81c784 100%); border-radius: 24px;
  padding: 1.6rem 2rem; margin-bottom: 1rem; box-shadow: 0 10px 24px rgba(27,94,32,.25); }
.hero h1 { color: #fff !important; font-size: 2rem; margin: 0 0 .4rem 0; padding: 0; line-height: 1.5; }
.hero p { color: #f1f8e9 !important; font-size: 1.02rem; line-height: 1.9; margin: 0; }
.hero .badge { display: inline-block; background: rgba(255,255,255,.22); color: #fff; padding: .15rem .9rem;
  border-radius: 999px; font-size: .85rem; margin-bottom: .6rem; }
.placeholder { border: 2px dashed #a5d6a7; border-radius: 16px; padding: 2rem 1rem; text-align: center;
  color: #4b6b55; background: #fbfef9; }
[data-testid="stFileUploaderDropzone"] { background: #fff; border: 2px dashed #66bb6a; border-radius: 16px; }
[data-testid="stFileUploaderDropzoneInstructions"] * { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content: "اسحب الصورة وأفلتها هنا\A JPG أو JPEG أو PNG";
  white-space: pre-line; display: block; text-align: center; color: #2e5d3a; }
[data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button * { display: none !important; }
[data-testid="stFileUploaderDropzone"] button::after { content: "اختيار صورة"; font-size: 1rem; }
/* Clearer touch-friendly controls and calmer result cards */
.stButton > button, [data-testid="stDownloadButton"] button {
  border-radius: 12px; min-height: 2.8rem; font-weight: 700; border: 1px solid #c8e6c9;
  transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button[kind="primary"] { background: #2e7d32; border-color: #2e7d32; color: #fff; }
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  box-shadow: 0 4px 12px rgba(27,94,32,.13); transform: translateY(-1px);
}
button[role="tab"] { font-weight: 700; padding-top: .7rem; padding-bottom: .7rem; }
button[role="tab"][aria-selected="true"] { color: #1b5e20; border-bottom-color: #2e7d32; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stExpander"] { border: 1px solid #dcebd9; border-radius: 12px; background: rgba(255,255,255,.8); }
[data-testid="stMetric"] { background: #fff; border: 1px solid #e0eee0; padding: .8rem 1rem; border-radius: 14px; }
@media (max-width: 700px) {
  .block-container { padding-left: .8rem; padding-right: .8rem; padding-top: .7rem; }
  .hero { padding: 1.2rem 1rem; border-radius: 18px; }
  .hero h1 { font-size: 1.55rem; }
  .hero p { font-size: .95rem; }
}
</style>"""

HERO_HTML = """<div class="hero">
<span class="badge">نسخة تطويرية للبحث العلمي</span>
<h1>🌿 Smart Farm AI</h1>
<p>تعرّف على النبات من صورة ورقته، وراجع الإرشادات الموثّقة المتاحة، واطّلع على مستشار المحاصيل حسب الموقع والموسم. كل النتائج تجريبية ولم تُقَس دقتها على صورنا بعد، وتظهر معها حدودها بوضوح.</p>
</div>"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Image validation (kept from Phase 1 / 2)
# ---------------------------------------------------------------
def load_image(uploaded_file):
    """Return (image, error_message). Exactly one of them is None."""
    data = uploaded_file.getvalue()
    if len(data) == 0:
        return None, "الملف فارغ. يرجى اختيار صورة أخرى."
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, f"حجم الصورة كبير جدًا. الحد الأقصى {MAX_FILE_MB} ميجابايت."
    try:
        with Image.open(BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()
        if image_format not in ALLOWED_FORMATS:
            return None, "نوع الملف غير مدعوم. يرجى رفع صورة JPG أو JPEG أو PNG."
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image, None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None, "تعذّرت قراءة الصورة. قد يكون الملف تالفًا، يرجى تجربة صورة أخرى."


# ---------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle():
    return pm.load_model()


@st.cache_resource(show_spinner=False)
def get_tables():
    return {
        "salinity": ca.load_salinity(),
        "perennial_kc": ca.load_perennial_kc(),
        "care": ca.load_care_notes(),
        "fao56": ca.load_fao56_rows(),
    }


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def cached_forecast(lat, lon):
    return wx.fetch_forecast(lat, lon)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def cached_history(lat, lon):
    return wx.fetch_climate_history(lat, lon)


def name_pair(key, index):
    ar, en = pm.display_names(key, index)
    return f"{ar} ({en})" if ar != en else ar


# ---------------------------------------------------------------
# Tab 1: plant identification
# ---------------------------------------------------------------
def show_identification(result, decision, catalog, index, core):
    status = decision["status"]
    key = decision["key"]
    pair = name_pair(key, index)
    percent = f"{decision['score'] * 100:.0f}%"

    if status == "likely":
        st.success(f"**النتيجة المحتملة: {pair}**")
    elif status == "uncertain":
        st.warning(f"**نتيجة غير مؤكدة:** تبدو الورقة أقرب إلى {pair}، لكن الأدلة غير كافية. لا تعتمد على هذه النتيجة.")
    elif status == "other":
        st.warning(
            f"**الصورة أقرب إلى: {pair}.** هذا النوع خارج النباتات الأساسية للمشروع "
            "(العنب والجوافة والكمثرى)، وقدرة النموذج عليه لم تُختبر عندنا بعد."
        )
    elif status == "not_plant":
        st.warning("**لا تبدو الصورة ورقة نبات.** يرجى رفع صورة ورقة واضحة.")
    else:
        st.warning("**لا يمكن تحديد النبات بثقة.** الأدلة غير كافية. جرّب صورة أوضح لورقة واحدة وبإضاءة جيدة.")

    st.metric("درجة تفضيل النموذج لأعلى احتمال", percent)
    st.caption(
        "هذه الدرجة ليست نسبة دقة. هي تفضيل النموذج بين الخيارات المعروضة له، ومجموعها 100% دائمًا حتى لو كانت الصورة غريبة. "
        "الدقة الحقيقية تُقاس بالتجارب على صور مصنّفة (evaluate_plants.py)."
    )

    st.markdown("**أعلى ثلاثة احتمالات:**")
    for k, score in result["ranked"][:3]:
        st.progress(min(max(score, 0.0), 1.0), text=f"{name_pair(k, index)}: {score * 100:.0f}%")

    with st.expander("تفاصيل تقنية"):
        st.write(
            f"النموذج: {pm.MODEL_NAME} (بدون تدريب على صورنا). الهامش عن ثاني احتمال: {decision['margin']:.2f}. "
            f"أعلى تشابه خام: {result['max_similarity']:.3f}. "
            f"حدود القرار (مختارة يدويًا وغير مقاسة): عالٍ ≥ {pm.HIGH_CONFIDENCE}، أدنى < {pm.LOW_CONFIDENCE}، هامش ≥ {pm.MIN_MARGIN}."
        )

    st.info(
        "حدود التعرف من الورقة وحدها: أنواع كثيرة وأصناف متقاربة تتشابه أوراقها، وصور الحقل تختلف عن الصور المرجعية. "
        "التعرف الأدق يحتاج أيضًا الثمرة والزهرة وشكل الشجرة. اعتبر النتيجة اقتراحًا يحتاج تأكيدًا."
    )


def show_species_facts(key, index):
    tables = get_tables()
    facts = ca.species_facts(key, tables["salinity"], tables["perennial_kc"], tables["care"])
    if key not in index:
        return
    ar, en = pm.display_names(key, index)
    st.markdown(f"#### معلومات موثّقة عن {ar} ({en})")
    sal = facts["salinity"]
    shown = False
    if sal["available"]:
        shown = True
        text = f"**تحمّل الملوحة (FAO):** {sal['rating_ar']}"
        if sal["estimate"]:
            text += " (تقدير وصفي)"
        if sal["threshold"] is not None:
            text += f" — حد تحمل التربة ECe ≈ {sal['threshold']} dS/m، وبعده يقل المحصول بنحو {sal['slope']}% لكل 1 dS/m."
        st.markdown(text)
        if sal["note"]:
            st.caption(f"ملاحظة المصدر: {sal['note']}")
    if facts["kc"]:
        shown = True
        kc = facts["kc"]["kc"]
        st.markdown(f"**معامل المحصول Kc (FAO-56):** بداية {kc[0]} / منتصف {kc[1]} / نهاية {kc[2]}. {facts['kc']['notes']}")
    elif key in ("guava",):
        st.markdown("**معامل المحصول Kc:** لا توجد قيمة للجوافة في جدول FAO-56، لذلك لا نعرض احتياجًا مائيًا لها.")
    for note in facts["care_notes"]:
        shown = True
        st.markdown(f"- {note['text_ar']}  \n  <small>المصدر: [{note['source']}]({note['source_url']})</small>", unsafe_allow_html=True)
    if not shown:
        st.caption("لا توجد معلومات موثّقة في قاعدة المعرفة الحالية لهذا النوع.")
    st.caption(
        "إرشادات الري الدقيق والتسميد والتقليم لا تُعرض إلا بعد إضافة مصدر موثوق ومراجعة مختص. "
        "المعلومات هنا عامة وليست وصفة لحقلك."
    )
    st.caption("مصادر: FAO Irrigation and Drainage Paper 61 (الملوحة)، FAO-56 (Kc)، وروابط المصادر في كل ملاحظة.")



def show_general_plant_guidance(key, index):
    """Show practical, conservative care guidance for every catalog species."""
    sp = index.get(key)
    if sp is None:
        return
    ar, en = pm.display_names(key, index)
    category = getattr(sp, "category", "other")

    # Crop-specific guidance is intentionally conservative and avoids invented
    # irrigation schedules, fertilizer rates, or pesticide prescriptions.
    profiles = {
        "grape": {
            "about": "العنب شجرة متسلقة/كرمة مثمرة، وتختلف احتياجاته حسب الصنف ونظام التربية والموسم.",
            "tips": [
                "وفّر ضوء الشمس والتهوية الجيدة، وتجنّب تزاحم الأفرع قدر الإمكان.",
                "اهتم بالصرف الجيد؛ لا تجعل الجذور واقفة في ماء راكد.",
                "تابع الأوراق والثمار بانتظام، خصوصًا ظهور مسحوق أبيض أو بقع زيتية وزغب أسفل الورقة.",
                "التقليم والتسميد وتوقيت الري يجب أن يتبعوا عمر الكرمة والصنف ومرحلة النمو، لا جدولًا واحدًا لكل الحقول."
            ],
        },
        "guava": {
            "about": "الجوافة شجرة مثمرة، وتتأثر جودة الإنتاج بظروف الري والتغذية والتقليم وصحة الشجرة.",
            "tips": [
                "وفّر ضوءًا مناسبًا وصرفًا جيدًا، وتجنّب التغريق المتكرر حول الجذور.",
                "راقب النموات الجديدة والوجهين العلوي والسفلي للأوراق، وافحص الثمار عند ظهور بقع أو تعفّن.",
                "لا تضف سمادًا بكميات كبيرة من غير معرفة عمر الشجرة وحالة التربة؛ تحليل التربة يساعد على تحديد الاحتياج.",
                "التقليم ومواعيد الري تختلف حسب الصنف والموسم والجو، فاستعن بتوصية محلية مناسبة."
            ],
        },
        "pear": {
            "about": "الكمثرى شجرة فاكهة، وقد تختلف ملاءمة الأصناف واحتياجات البرودة والتلقيح باختلاف المنطقة.",
            "tips": [
                "تأكد أن الصنف مناسب لمناخ منطقتك، وأن احتياجات التلقيح متوفرة إذا كان الصنف يحتاج ملقّحًا.",
                "حافظ على رطوبة تربة مناسبة من دون تغريق، واهتم بالصرف.",
                "راقب الأوراق والأفرع والثمار بانتظام، وسجّل متى بدأت الأعراض وأين ظهرت.",
                "التقليم والتسميد ومواعيد الري تعتمد على الصنف وعمر الشجرة والموسم؛ لا تعتمد على كمية ثابتة لكل الأشجار."
            ],
        },
    }
    profile = profiles.get(key)
    st.markdown("#### 🌳 معلومات ونصائح أولية عن النبات")
    st.markdown(f"**{ar} ({en})**")
    if getattr(sp, "scientific", None):
        st.caption("الاسم العلمي المسجّل: " + "، ".join(sp.scientific))
    if profile:
        st.write(profile["about"])
        tips = profile["tips"]
    elif category == "fruit":
        st.write("هذا النوع مُدرج في قاموس التعرف بوصفه محصولًا مثمرًا. المعلومات التالية عامة، وليست توصية خاصة بالصنف أو الحقل.")
        tips = [
            "تحقق من ملاءمة الصنف لمناخ منطقتك، ومن احتياجات التلقيح إن كانت تنطبق عليه.",
            "وفّر ضوءًا مناسبًا وصرفًا جيدًا، واضبط الري بحسب نوع التربة والطقس وعمر النبات بدلًا من جدول ثابت.",
            "افحص الأوراق والثمار والأفرع دوريًا، وسجّل الأعراض وتاريخ ظهورها للمساعدة في التشخيص.",
            "لتحسين الإثمار، لا تفترض أن زيادة السماد أو الماء أفضل؛ اعتمد على تحليل التربة وإرشاد زراعي محلي للتقليم والتغذية."
        ]
    elif category == "crop":
        st.write("هذا محصول زراعي، لكن طريقة رعايته تختلف كثيرًا حسب النوع والصنف وموعد الزراعة والمنطقة.")
        tips = [
            "استخدم بذورًا أو شتلات سليمة ومناسبة للصنف والمنطقة، واتبع موعد الزراعة المحلي.",
            "اضبط الري حسب مرحلة النمو ورطوبة التربة والطقس، وتجنب الإفراط أو التعطيش الشديد.",
            "راقب الآفات والأعراض مبكرًا، وتجنب استخدام مبيد قبل التأكد من سبب المشكلة.",
            "اختبر التربة إن أمكن، ولا تحدد جرعات السماد أو مواعيد المكافحة من دون بيانات موثوقة للمحصول."
        ]
    else:
        st.write("لم تتوفر في قاعدة المعرفة الحالية بطاقة رعاية خاصة بهذا النوع؛ لذلك نعرض إرشادات عامة فقط.")
        tips = [
            "تأكد من الاسم النباتي قبل اتخاذ قرار زراعي مهم؛ فقد تتشابه أوراق أنواع مختلفة.",
            "راقب ضوء المكان ورطوبة التربة والصرف، وتجنب الري الزائد أو ترك النبات يعاني من العطش.",
            "افحص النبات بانتظام وصوّر الأعراض بوضوح إذا ظهرت، من دون استخدام علاج كيميائي بالتخمين.",
            "للتقليم والتسميد ومكافحة الآفات، استخدم إرشادات خاصة بالنوع والصنف ومن مصدر زراعي موثوق."
        ]
    for tip in tips:
        st.markdown(f"- {tip}")
    st.info("لإعطاء نصيحة أدق، يحتاج التطبيق مستقبلًا إلى معرفة عمر النبات، والصنف، وموقع المزرعة، ونوع التربة، ومرحلة النمو. هذه النصائح عامة ولا تضمن زيادة المحصول وحدها.")


def show_disease_panel(key, decision, result):
    st.markdown("#### الأعراض المحتملة (كشف الأمراض)")
    head = disease.DiseaseHead.load(key)
    if head is None:
        st.info(
            "لا يوجد نموذج أمراض مدرَّب ومُختبَر لهذا النبات في المشروع حتى الآن، "
            "لذلك لا يعرض التطبيق أي مرض محتمل. (للتدريب: disease.py، وانظر README_AR.md)"
        )
        return
    if decision["status"] != "likely" or decision["key"] != key:
        st.warning("لن نعرض أعراضًا قبل التأكد من نوع النبات بدرجة كافية.")
        return
    outcome = head.interpret(result["embedding"])
    prob = f"{outcome['probability'] * 100:.0f}%"
    if outcome["status"] == "healthy":
        st.success(f"لا تظهر أعراض من الأنواع المُدرَّب عليها ({prob}). هذا لا يعني أن النبات سليم تمامًا.")
    elif outcome["status"] == "possible_symptoms":
        st.warning(f"أعراض قد تتوافق مع: **{outcome['top']}** ({prob}). هذا احتمال وليس تشخيصًا مؤكدًا.")
    else:
        st.warning("لا يستطيع النموذج تحديد الحالة بثقة. الأعراض غير كافية للتشخيص.")
    test = head.metrics.get("test", {})
    if test:
        st.caption(
            f"أداء هذا النموذج على صور اختبار لم يرها ({test.get('n')} صورة، من مجموعة بيانات التدريب نفسها): "
            f"الدقة {test.get('accuracy', 0):.2f}، macro-F1 {test.get('macro_f1', 0):.2f}. "
            "لم يُختبر على صور الحقل بعد. الأمراض غير المُدرَّب عليها لن تظهر. استشر مهندسًا زراعيًا."
        )
    else:
        st.caption("لا توجد أرقام اختبار محفوظة لهذا النموذج، فاعتبره غير موثوق.")


def plant_tab():
    catalog = pm.load_catalog()
    index = pm.species_index(catalog)
    core = set(pm.core_keys(catalog))

    col_in, col_out = st.columns(2, gap="large")
    image, error_message, source_id = None, None, None

    with col_in:
        st.subheader("ارفع صورة الورقة")
        uploaded = st.file_uploader("صورة الورقة", type=ALLOWED_TYPES, label_visibility="collapsed")
        st.caption(f"JPG أو JPEG أو PNG، حتى {MAX_FILE_MB} ميجابايت. من الهاتف: اختر «التقاط صورة» من نافذة الاختيار.")
        shot = None
        if st.checkbox("استخدام كاميرا المتصفح (قد لا يعمل على عنوان http داخل الشبكة)"):
            shot = st.camera_input("التقط صورة", label_visibility="collapsed")
        chosen = uploaded or shot
        if chosen is not None:
            image, error_message = load_image(chosen)
            if error_message:
                st.error(error_message)
            else:
                source_id = hashlib.sha256(chosen.getvalue()).hexdigest()

        extra_text = st.text_input(
            "اسم علمي إضافي للمقارنة (اختياري)",
            placeholder="مثال: Ficus sycomorus , Annona muricata",
            help="يضاف إلى القائمة لهذه الصورة فقط، بدون تعديل أي ملف. افصل الأسماء بفاصلة.",
        )
        extra = [t.strip() for t in extra_text.split(",") if t.strip()][:5]
        with st.expander("نصائح لتصوير أفضل"):
            st.markdown(
                "- صوّر ورقة واحدة واضحة وقريبة، في ضوء النهار.\n"
                "- اجعل الورقة في المنتصف وبلا حركة، وخلفية بسيطة.\n"
                "- صوّر وجه الورقة وظهرها إن أمكن."
            )

    with col_out:
        st.subheader("معاينة الصورة")
        if image is None:
            st.markdown('<div class="placeholder">🍃<br>ستظهر معاينة الصورة هنا بعد رفعها</div>', unsafe_allow_html=True)
            if error_message is None:
                st.info("ارفع صورة ورقة ليبدأ التعرف على النبات.")
            return
        preview = image.copy()
        preview.thumbnail((1000, 1000))
        st.image(preview, caption=f"{image.width} × {image.height} بكسل")
        for note in pm.image_quality_notes(image):
            st.warning(note)

        analysis_id = f"{source_id}|{'|'.join(extra)}"
        saved = st.session_state.get("analysis")
        if saved and saved["id"] == analysis_id:
            result = saved["result"]
        else:
            try:
                with st.spinner("جارٍ تحميل النموذج وتحليل الصورة... أول تشغيل قد يستغرق عدة دقائق (راقب نافذة الطرفية)."):
                    result = pm.classify(image, get_bundle(), extra_species=extra)
                st.session_state["analysis"] = {"id": analysis_id, "result": result}
            except ImportError as exc:
                st.error("مكتبات النموذج غير مثبّتة بعد. شغّل هذا الأمر في PowerShell ثم أعد تشغيل الموقع:")
                st.code(INSTALL_COMMAND, language="powershell")
                with st.expander("تفاصيل تقنية"):
                    st.code(f"{type(exc).__name__}: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                st.error("تعذّر تحميل النموذج أو تشغيله. تأكد من الإنترنت عند أول تشغيل ومن مساحة القرص، ثم أعد المحاولة.")
                with st.expander("تفاصيل تقنية"):
                    st.code(f"{type(exc).__name__}: {exc}")
                return

        decision = pm.decide(result["ranked"], result["max_similarity"], core)
        if decision["status"] in ("likely", "uncertain") and decision["key"] in core:
            st.session_state["identified_core"] = decision["key"]
        with st.container(border=True):
            st.subheader("نتيجة التعرف")
            show_identification(result, decision, catalog, index, core)
        if decision["status"] in ("likely", "uncertain", "other") and decision["key"] in index:
            with st.container(border=True):
                show_species_facts(decision["key"], index)
                show_general_plant_guidance(decision["key"], index)
        if decision["status"] in ("likely", "uncertain") and decision["key"] in core:
            with st.container(border=True):
                show_disease_panel(decision["key"], decision, result)


# ---------------------------------------------------------------
# Tab 2: Smart Crop Advisor
# ---------------------------------------------------------------
def advisor_inputs():
    st.markdown("##### 1) الموقع")
    choice = st.selectbox("المحافظة", GOVERNORATES + [OTHER_PLACE])
    query = st.text_input("اكتب اسم المنطقة أو المدينة", value="" if choice == OTHER_PLACE else choice,
                          help="مثال: المنصورة أو سيوة أو الخارجة. كلما كان الاسم أدق كان تحديد الموقع أدق.")
    st.caption("تحديد الموقع التلقائي من الجهاز غير مفعّل: المتصفح يشترطه اتصالًا آمنًا (HTTPS)، فنكتفي بالاسم المكتوب حاليًا.")

    st.markdown("##### 2) الملوحة (مهمة جدًا في الأراضي المصرية)")
    mode = st.radio("ما المعلوم لديك؟", ["غير معروفة", "ملوحة التربة ECe (dS/m)", "ملوحة ماء الري EC (dS/m)"], horizontal=True)
    value = None
    if mode != "غير معروفة":
        value = st.number_input("القيمة (dS/m)", min_value=0.0, max_value=40.0, value=2.0, step=0.1)
    ece = None
    if mode.startswith("ملوحة التربة"):
        ece = value
    elif mode.startswith("ملوحة ماء"):
        ece = ca.ece_from_ecw(value)
        st.caption(f"تقدير تقريبي لملوحة التربة من ماء الري: ECe ≈ {ece:.1f} dS/m (معامل {ca.ECE_FROM_ECW_FACTOR}). القياس المباشر للتربة أدق.")

    st.markdown("##### 3) معلومات أخرى (لا تدخل في الترتيب حاليًا)")
    soil = st.selectbox("قوام التربة", ["غير معروف", "رملية", "طمية", "طينية", "جيرية"])
    drainage = st.selectbox("الصرف", ["غير معروف", "جيد", "متوسط", "ضعيف"])
    ph_known = st.checkbox("أعرف درجة حموضة التربة (pH)")
    return query.strip(), ece, soil, drainage, ph_known


def show_candidate(rec):
    with st.container(border=True):
        when = "هذا الشهر" if rec["distance"] == 0 else "الشهر القادم"
        st.markdown(f"**{rec['name_ar']}** ({rec['name_en']}) — موعد زراعة مذكور في FAO-56: **{when}** ({ca.MONTH_NAMES_AR[rec['start_month'] - 1]})")
        st.caption(f"صف المصدر: منطقة «{rec['region_label']}» ({rec['climate_class']})، طول الموسم ≈ {rec['cycle_days']} يومًا.")
        water = rec["water"]
        if water:
            st.markdown(
                f"احتياج الماء التقريبي للموسم: ETc ≈ **{water['etc_mm']:.0f} مم**، أمطار مناخية ≈ {water['rain_mm']:.0f} مم، "
                f"الفرق التقريبي ≈ **{water['net_mm']:.0f} مم** (بدون حساب فواقد الري)."
            )
        sal = rec["salinity"]
        if sal["available"]:
            line = f"تحمّل الملوحة: {sal['rating_ar']}"
            if sal["yr"] is not None:
                line += f" — المحصول النسبي المتوقع عند الملوحة المدخلة ≈ **{sal['yr']:.0f}%** (نموذج FAO)"
            st.markdown(line)
        for flag in rec["flags"]:
            st.caption("⚠️ " + flag)
        if rec["notes"]:
            st.caption("ملاحظة الصف: " + rec["notes"])


def advisor_tab():
    st.subheader("📍 مستشار المحاصيل والظروف المحلية")
    st.info(
        "أدخل اسم منطقتك في مصر أو خارجها للحصول على مؤشرات من الطقس والمناخ ومراجع FAO. "
        "يغطي جدول مواعيد الزراعة الحالي 23 محصولًا، لكن توافر بيانات الري والملوحة يختلف من محصول لآخر. "
        "لا يتوقع الإنتاجية أو الأرباح، ولا يستخدم تحليل التربة والصرف ودرجة الحموضة بصورة مكتملة؛ لذلك لا تعتبر النتيجة قرار زراعة نهائيًا."
    )
    query, ece, soil, drainage, ph_known = advisor_inputs()
    if not st.button("حلّل الموقع", type="primary"):
        saved = st.session_state.get("advisor")
        if saved:
            render_advice(saved)
        return
    if not query:
        st.warning("اكتب اسم المنطقة أولًا.")
        return
    try:
        with st.spinner("جارٍ جلب بيانات الطقس والمناخ..."):
            places = wx.geocode(query)
            if not places:
                st.error("لم أجد هذا الموقع. جرّب اسمًا آخر أو أدق.")
                return
            place = places[0]
            forecast = cached_forecast(place["latitude"], place["longitude"])
            history = cached_history(place["latitude"], place["longitude"])
    except wx.NETWORK_ERRORS as exc:
        st.error("تعذّر الوصول إلى خدمة الطقس. لن نعرض بيانات قديمة على أنها حالية. تأكد من الإنترنت ثم أعد المحاولة.")
        with st.expander("تفاصيل تقنية"):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    normals = wx.monthly_normals(history["daily"])
    advice = ca.advise(normals, date.today(), ece, fao_rows=get_tables()["fao56"], salinity=get_tables()["salinity"])
    st.session_state["advisor"] = {
        "place": place, "others": places[1:4], "forecast": forecast, "history": history, "normals": normals,
        "alerts": wx.general_alerts(forecast), "advice": advice, "ece": ece,
        "gaps": ca.missing_information(ece is not None, soil, drainage, ph_known),
    }
    render_advice(st.session_state["advisor"])


def render_advice(data):
    place = data["place"]
    st.markdown(f"### 📍 {place['name']}، {place['admin1']} ({place['latitude']:.2f}, {place['longitude']:.2f})")
    if data["others"]:
        st.caption("نتائج أخرى للاسم نفسه: " + "، ".join(f"{p['name']} ({p['admin1']})" for p in data["others"]) + ". إن لم يكن موقعك فاكتب اسمًا أدق.")
    current = data["forecast"].get("current", {})
    st.markdown(
        f"**الطقس الآن:** {current.get('temperature_2m', '؟')}°م، رطوبة {current.get('relative_humidity_2m', '؟')}%. "
        f"<small>(جُلبت: {data['forecast']['fetched_at']})</small>",
        unsafe_allow_html=True,
    )
    for alert in data["alerts"]:
        st.warning(alert)
    if not data["alerts"]:
        st.success("لم تُرصد في التوقعات حالات تتجاوز حدود التنبيه العامة المستخدمة (وهي تقريبية).")

    with st.expander("المناخ المحلي (متوسطات من بيانات تاريخية)"):
        rows = ["| الشهر | صغرى | عظمى | ET0 يوميًا (مم) | أمطار الشهر (مم) | أيام صقيع/سنة |", "|---|---|---|---|---|---|"]
        for m, n in sorted(data["normals"].items()):
            rows.append(
                f"| {ca.MONTH_NAMES_AR[m - 1]} | {n['tmin_mean']:.1f} | {n['tmax_mean']:.1f} | {n['et0_mean_daily']:.1f} | "
                f"{n['rain_total_mm']:.1f} | {n['frost_days_per_year']:.1f} |"
            )
        st.markdown("\n".join(rows))
        st.caption(f"الفترة: {data['history']['period']} — جُلبت: {data['history']['fetched_at']}.")

    advice = data["advice"]
    st.markdown("### المحاصيل المرشحة لهذا الوقت")
    if data["ece"] is not None:
        st.caption(f"الترتيب: الأقرب موعدًا أولًا، ثم الأعلى محصولًا نسبيًا متوقعًا عند ECe ≈ {data['ece']:.1f} dS/m.")
    else:
        st.caption("الترتيب: الأقرب موعدًا ثم الاسم. أدخل الملوحة لترتيب أدق.")
    if not advice["candidates"]:
        st.warning("لا توجد محاصيل في الجدول المرجعي موعد زراعتها هذا الشهر أو القادم.")
    for rec in advice["candidates"]:
        show_candidate(rec)

    with st.expander(f"محاصيل خارج الموسم الحالي ({len(advice['out_of_season'])})"):
        for rec in advice["out_of_season"]:
            st.markdown(f"- {rec['name_ar']}: أقرب موعد في FAO-56 بعد {rec['distance']} أشهر ({ca.MONTH_NAMES_AR[rec['start_month'] - 1]})")

    st.markdown("### معلومات ناقصة تحدّ من دقة التوصية")
    for gap in data["gaps"]:
        st.markdown(f"- {gap}")
    st.caption(
        "قد توجد قيود رسمية على بعض المحاصيل كثيفة المياه أو على مواعيد زراعتها؛ راجع مديرية الزراعة المحلية قبل القرار. "
        "القيم المرجعية استرشادية، وتحتاج مراجعة الإرشاد الزراعي المحلي."
    )
    st.markdown(
        f"مصادر: [Open-Meteo]({wx.ATTRIBUTION_URL}) (طقس ومناخ)، FAO Irrigation and Drainage Paper 56 (مواعيد الزراعة وKc)، "
        "FAO Irrigation and Drainage Paper 61 (الملوحة).",
    )



# ---------------------------------------------------------------
# Tab: guided symptom check (questions and options)
# ---------------------------------------------------------------
KIND_ICON = {"cultural": "🌱", "sanitation": "✂️", "timing": "🗓️", "chemical_note": "⚠️", "info": "ℹ️", "none": "•"}


@st.cache_resource(show_spinner=False)
def get_kb():
    return dg.load_kb()


def show_disease_card(entry, rank_label):
    disease_ = entry["disease"]
    kb = get_kb()
    with st.container(border=True):
        kind = " (حالة تشبه المرض وليست مرضًا مؤكدًا)" if disease_["kind"] == "lookalike" else ""
        st.markdown(f"**{rank_label}: {disease_['name_ar']}** ({disease_['name_en']}){kind}")
        st.caption("ترتيب نوعي حسب تطابق الأعراض، وليس نسبة احتمال أو درجة دقة.")
        st.markdown(disease_["summary_ar"])
        if entry["supports"]:
            st.markdown("**علامات تدعم هذا الاحتمال:**")
            for qid, options in entry["supports"]:
                st.markdown(f"- {dg.question_text(kb, qid)} ← {'، '.join(dg.option_text(kb, qid, o) for o in options)}")
        if entry["contradicts"]:
            st.markdown("**علامات لا تتفق معه:**")
            for qid, options in entry["contradicts"]:
                st.markdown(f"- {dg.question_text(kb, qid)} ← {'، '.join(dg.option_text(kb, qid, o) for o in options)}")
        if entry["not_yet_observed"]:
            st.caption("علامات لم تُفحص بعد: " + "؛ ".join(dg.question_text(kb, q) for q in entry["not_yet_observed"]))
        st.markdown(f"**كيف تفرّق بينه وبين الحالات المشابهة؟** {disease_['how_to_tell_ar']}")
        with st.expander("🌱 التعامل والعلاج الآمن — إرشادات من المصادر"):
            if disease_["kind"] == "lookalike":
                st.warning("هذه حالة قد تشبه المرض وليست مرضًا مؤكدًا. لا تبدأ علاجًا كيميائيًا قبل التأكد من السبب.")
            management = disease_.get("management", [])
            if not management or all(item.get("kind") == "none" for item in management):
                st.info("لا توجد توصية إدارة موثقة لهذه الحالة في قاعدة المعرفة الحالية. اطلب تقييم مهندس زراعي قبل المعالجة.")
            else:
                st.markdown("**ما تقوله المصادر المتاحة:**")
                for item in management:
                    icon = KIND_ICON.get(item.get("kind"), "•")
                    label = {
                        "cultural": "ممارسة زراعية",
                        "sanitation": "نظافة وإزالة مصادر العدوى",
                        "timing": "توقيت الفحص أو التدخل",
                        "chemical_note": "تنبيه بخصوص المكافحة الكيميائية",
                        "info": "معلومة مساعدة",
                        "none": "لا توجد توصية",
                    }.get(item.get("kind"), "إرشاد")
                    tail = f"  \n  <small>المصدر: [{item['source']}]({item['url']})</small>" if item.get("url") else ""
                    st.markdown(f"- {icon} **{label}:** {item['text_ar']}{tail}", unsafe_allow_html=True)
            st.warning("قبل استخدام أي مبيد: أكّد التشخيص مع مهندس زراعي، وتحقق من أن المنتج مسجّل في مصر لهذا المحصول والمرض، واتبع ملصق المنتج وفترة ما قبل الحصاد ووسائل الوقاية. لا يقدّم التطبيق أسماء مبيدات أو جرعات.")
            st.caption("هذه إرشادات أولية وليست وصفة علاجية. بعض الحالات غير مدرجة، وقاعدة المعرفة لم يراجعها مهندس زراعي مصري بعد.")
        with st.expander("المصادر العلمية"):
            for source in disease_["sources"]:
                st.markdown(f"- [{source['name']}]({source['url']})")


def symptom_tab():
    kb = get_kb()
    st.subheader("🩺 فحص الأعراض والتعامل معها")
    st.info(
        "اختر المحصول وأجب عن العلامات التي تراها فعلًا. سيعرض التطبيق الحالات الموجودة في قاعدة المعرفة، "
        "وما يدعم كل احتمال، والإرشادات المنشورة المرتبطة به. النتيجة ليست تشخيصًا نهائيًا. "
        "التغطية الحالية محدودة بـ12 حالة للعنب والجوافة والكمثرى؛ إذا كان محصولك غير موجود، فلا تستخدم هذه النتيجة لتشخيصه أو علاجه."
    )
    st.caption(kb["note_ar"])

    plants = kb["plants"]
    keys = list(plants)
    other_key = "other_crop_not_supported"
    options = keys + [other_key]
    labels = {**plants, other_key: "محصول آخر غير موجود في القائمة"}
    preset = st.session_state.get("identified_core")
    plant = st.selectbox("نوع النبات", options, index=keys.index(preset) if preset in keys else 0, format_func=labels.get)
    if preset in keys:
        st.caption("اخترنا النبات تلقائيًا من آخر صورة جرى التعرف عليها. غيّره إن لم يكن صحيحًا.")
    if plant == other_key:
        crop_name = st.text_input("اكتب اسم المحصول", placeholder="مثال: طماطم، مانجو، قمح...")
        st.warning(
            "قاعدة الأمراض الحالية لا تحتوي بيانات موثّقة لهذا المحصول، لذلك لن نخمن المرض أو العلاج. "
            "صوّر النبات كاملًا والورقة من الوجهين، وسجّل مكان ظهور الأعراض وتاريخ بدايتها، "
            "وتجنّب خلط أو رش مبيد قبل مراجعة مهندس زراعي أو وحدة الإرشاد الزراعي."
        )
        st.info("نريد توسيع قاعدة المعرفة لتشمل محاصيل أكثر، لكن يجب إضافة كل محصول مع مصادر موثوقة ومراجعة مختص قبل إصدار توصيات علاجية له.")
        return

    questions = dg.applicable_questions(kb, plant)
    reset_version = st.session_state.get(f"sym_reset_version_{plant}", 0)
    # Ask three essential questions first. If the evidence is unclear, offer an
    # additional discriminating question rather than overwhelming the user.
    initial_questions = questions[:3]
    answers = {}
    for question in initial_questions:
        ids = [o["id"] for o in question["options"]]
        labels = {o["id"]: o["text_ar"] for o in question["options"]}
        widget_key = f"sym_{plant}_{question['id']}" if reset_version == 0 else f"sym_{plant}_{question['id']}_{reset_version}"
        if question["type"] == "multi":
            answers[question["id"]] = st.multiselect(
                question["text_ar"], ids, format_func=labels.get, key=widget_key,
                help="اختر كل ما ينطبق على النبات. إذا لم تكن متأكدًا، اتركه فارغًا."
            )
        else:
            answers[question["id"]] = [st.selectbox(
                question["text_ar"], ids, format_func=labels.get, key=widget_key
            )]

    # Add previously submitted follow-up answers for this plant.
    followup_key = f"sym_followup_answers_{plant}_{reset_version}"
    saved_followups = st.session_state.get(followup_key, {})
    answers.update(saved_followups)

    ranked = dg.score_diseases(kb, plant, answers)
    outcome = dg.interpret(ranked, answers)
    status = outcome["status"]
    st.markdown("### النتيجة الأولية")
    if status == "insufficient":
        st.warning(f"المعلومات غير كافية حتى الآن. جاوب عن {dg.MIN_ANSWERS} أسئلة على الأقل، واختار «لا أعرف» أو اترك السؤال بدون إجابة لو مش متأكد.")
    elif status == "no_symptoms":
        st.success("لم تذكر أعراضًا ظاهرة. هذا لا يؤكد أن النبات سليم؛ راقبه وكرر الفحص إن ظهر جديد.")
    elif status == "no_match":
        st.warning("الإجابات لا تتطابق بدرجة كافية مع الحالات الموجودة. قد تكون الحالة غير مدرجة، لذلك لن نخمن مرضًا أو علاجًا.")
    elif status == "ambiguous":
        st.warning("هناك أكثر من احتمال متقارب. سنسألك سؤالًا إضافيًا يفرّق بين الاحتمالات بدلًا من اختيار مرض بالتخمين.")
    else:
        st.success("يوجد احتمال متقدم حسب إجاباتك، لكنه ليس تشخيصًا مؤكدًا.")

    if status in ("possible", "ambiguous"):
        shown = [outcome["top"]] + outcome.get("close", [])
        for i, entry in enumerate(shown, 1):
            show_disease_card(entry, f"الاحتمال {i}")
        others = [r for r in ranked if r not in shown and r["evidence"] > 0]
        if others:
            with st.expander(f"احتمالات أضعف ({len(others)})"):
                for entry in others:
                    show_disease_card(entry, "احتمال أضعف")

    # When the first answers are ambiguous or fail to match, ask the most useful
    # remaining question. Answers are stored separately so the UI stays simple.
    suggestion = dg.suggest_next_question(kb, plant, answers)
    if suggestion and status in ("ambiguous", "no_match"):
        q = suggestion[0]
        # Do not ask again about one of the first three questions if the user
        # explicitly selected "unknown" there; move to a genuinely new question.
        initial_ids = {item["id"] for item in initial_questions}
        if q["id"] in initial_ids:
            answered_followups = set(saved_followups)
            alternatives = [item for item in questions if item["id"] not in initial_ids and item["id"] not in answered_followups]
            if alternatives:
                q = alternatives[0]
        qid = q["id"]
        st.markdown("### سؤال إضافي لتحديد الاحتمال بشكل أفضل")
        st.write(q["text_ar"])
        ids = [o["id"] for o in q["options"]]
        labels = {o["id"]: o["text_ar"] for o in q["options"]}
        extra_key = f"sym_followup_choice_{plant}_{qid}_{reset_version}"
        selected = st.selectbox(
            "اختر أقرب وصف لما تراه", [dg.UNKNOWN] + [option_id for option_id in ids if option_id != dg.UNKNOWN],
            format_func=lambda v: "لم أتأكد / لا أعرف" if v == dg.UNKNOWN else labels.get(v, v),
            key=extra_key,
        )
        if st.button("إضافة الإجابة وتحسين الترجيح", key=f"sym_followup_submit_{plant}_{qid}_{reset_version}"):
            if selected != dg.UNKNOWN:
                saved = dict(st.session_state.get(followup_key, {}))
                saved[qid] = [selected]
                st.session_state[followup_key] = saved
                st.success("تمت إضافة الإجابة. راجع النتيجة المحدّثة في هذه الصفحة بعد إعادة التشغيل.")
                if hasattr(st, "rerun"):
                    st.rerun()
            else:
                st.info("اختَر وصفًا تعرفه فعلًا، أو اترك السؤال إذا لم تكن متأكدًا.")

    if st.button("بدء فحص جديد لهذا المحصول", key=f"sym_reset_{plant}_{reset_version}"):
        # Versioned widget keys reset inputs without mutating active widget state.
        st.session_state[f"sym_reset_version_{plant}"] = reset_version + 1
        if hasattr(st, "rerun"):
            st.rerun()

    st.caption(
        "للتأكد: اعرض العينة أو الصور على مهندس زراعي أو معمل أمراض نبات قبل أي معاملة كيميائية. "
        "الأمراض الأخرى الشائعة في مصر وغير المدرجة هنا لن تظهر في النتيجة."
    )


# ---------------------------------------------------------------
# Tab 3: science and limits
# ---------------------------------------------------------------
def science_tab():
    st.subheader("🔬 العلم والحدود")
    st.markdown(
        """
### للمستخدم: ماذا يستطيع التطبيق أن يفعل؟
- **التعرّف على النبات:** يقارن صورة الورقة بقائمة الأنواع الموجودة في النظام. النتيجة اقتراح أولي وليست إثباتًا نهائيًا لهوية النبات.
- **فحص الأمراض:** يسألك عن علامات واضحة باختيارات بسيطة، ويرتّب الحالات التي تتفق مع إجاباتك. إذا لم تكفِ المعلومات أو تقاربت الاحتمالات، يطلب معلومات إضافية أو يوضح أنه لا يستطيع الترجيح.
- **إرشادات التعامل:** يعرض الممارسات الزراعية الموجودة في قاعدة المعرفة ومصادرها. لا يصف مبيدًا أو جرعة، ولا ينبغي استخدام علاج كيميائي اعتمادًا على نتيجة التطبيق وحدها.
- **مستشار المحاصيل:** يعرض مؤشرات الطقس والمناخ وبعض بيانات FAO، لكن التوصية لا تغني عن تحليل التربة والإرشاد الزراعي المحلي.

### حدود النسخة الحالية
| الجزء | الموجود حاليًا | ما الذي لم نثبته بعد؟ |
|---|---|---|
| التعرف على النبات | قاموس يضم نحو 54 مدخلًا ونموذج BioCLIP 2 جاهز | لم تُقَس الدقة على مجموعة صور مصرية مستقلة، ولا يدعم النظام كل الأنواع الموجودة في العالم. |
| فحص الأمراض | 12 حالة مرتبطة بالعنب والجوافة والكمثرى، مع أسئلة وإرشادات ومصادر | لا يغطي أمراض جميع المحاصيل، ولم تُقَس دقته على حالات مؤكدة ميدانيًا، والمصادر لم يراجعها مهندس زراعي مصري بعد. |
| مستشار المحاصيل | بيانات مواعيد زراعة لـ23 محصولًا، مع اختلاف توفر بيانات الماء والملوحة | لا يعطي توصية مكتملة لكل حقل، ولا يتنبأ بالإنتاج أو الأرباح. |
| نصائح الرعاية | إرشادات عامة وملاحظات موثقة لبعض الأنواع | لا يمكن ضمان زيادة المحصول دون بيانات الصنف والعمر والتربة والمناخ وتجارب ميدانية. |

### خطة التحقق العلمي
1. جمع صور حقيقية وتأكيد أسماء النباتات بواسطة شخص موثوق أو مختص.
2. اختبار النموذج على صور لم تُستخدم في تطويره، وتسجيل الدقة وPrecision وRecall وF1 لكل نوع.
3. اختبار فحص الأعراض على حالات أكدها مختص، وتسجيل الحالات الصحيحة والخاطئة والحالات التي امتنع فيها النظام عن الترجيح.
4. مراجعة الإرشادات الزراعية مع مهندس زراعي محلي قبل توسيع التوصيات العلاجية.
5. عرض النتائج الفعلية فقط، مع ذكر حجم العينة ومصدرها وحدودها.

**الإضافة البحثية المقترحة:** اختبار ما إذا كان طرح سؤال توضيحي عند تقارب الاحتمالات يحسن الترجيح، مع قياس عدد الأسئلة الإضافية ومقدار التغير في صحة النتيجة. هذه فرضية تحتاج تجربة، وليست نتيجة مثبتة حتى الآن.
        """
    )
    with st.expander("للمهتم بالتفاصيل التقنية"):
        st.markdown(
            f"- نموذج التعرف: `{pm.MODEL_NAME}`؛ نموذج جاهز لم يُدرّب على صور محلية.\n"
            "- فحص الأمراض يعتمد على قواعد مطابقة للأعراض، وليس نموذج رؤية حاسوبية مدرّبًا على صور الأمراض.\n"
            "- درجة تشابه نموذج النبات ليست احتمالًا إحصائيًا مؤكدًا.\n"
            "- الاختبارات الآلية تتحقق من سلوك الكود، لكنها لا تثبت الدقة الزراعية أو فاعلية أي علاج.\n"
            "- ينبغي توثيق أي مصدر خارجي أو مكتبة أو نموذج مستخدم في المشروع."
        )
    st.warning("هذا مساعد زراعي تجريبي، وليس بديلًا عن مهندس زراعي. عند وجود إصابة سريعة الانتشار أو تدهور واضح، اطلب تقييمًا محليًا. لا تستخدم مبيدًا أو جرعة اعتمادًا على نتيجة التطبيق وحدها.")


tab1, tab_sym, tab2, tab3 = st.tabs(["🌿 التعرف على النبات", "🩺 فحص الأعراض", "📍 مستشار المحاصيل", "🔬 العلم والحدود"])
with tab1:
    plant_tab()
with tab_sym:
    symptom_tab()
with tab2:
    advisor_tab()
with tab3:
    science_tab()

st.caption(
    "تُعالَج الصورة على جهازك ولا تُرسل لأي خدمة، ولا تُحفظ في ملف. تُرسل فقط اسم الموقع الذي تكتبه إلى Open-Meteo عند الضغط على «حلّل الموقع». "
    "يُنزَّل النموذج من الإنترنت مرة واحدة."
)
