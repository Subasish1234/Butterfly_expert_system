import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import wikipediaapi
import folium
from streamlit_folium import st_folium
import base64

# --- App Configuration ---
st.set_page_config(
    page_title="BioScan AI: Butterfly Expert System",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# ADVANCED STYLING AND BACKGROUND
# ==============================================================================

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("background.jpg")

page_style = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Poppins:wght@300;400&display=swap');

.stApp {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}
.glass-card {{
    background: rgba(25, 25, 40, 0.7);
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 2rem;
    margin-bottom: 2rem;
    animation: fadeIn 1s ease-out;
}}
h1 {{
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    text-align: center;
    font-size: 3rem;
    background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
h2, h3 {{
    font-family: 'Orbitron', sans-serif;
    color: #00C9FF;
}}
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(20px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ==============================================================================
# DATA AND MODEL LOADING
# ==============================================================================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('butterfly_expert_model.keras')
    df = pd.read_csv('master_labels.csv')
    all_species = sorted(df['species_name'].unique())
    id_to_species = {i: name for i, name in enumerate(all_species)}
    CONSERVATION_STATUS = {
        'Monarch': ('Endangered', '🔴'), 'Black Hairstreak': ('Near Threatened', '🟡'),
        'Adonis': ('Least Concern', '🟢'), 'Adonis Blue': ('Least Concern', '🟢'),
        'American Snout': ('Least Concern', '🟢'), 'An 88': ('Least Concern', '🟢'),
        'Banded Peacock': ('Least Concern', '🟢'), 'Beckers White': ('Least Concern', '🟢'),
        'Cabbage White': ('Least Concern', '🟢'), 'Chestnut': ('Least Concern', '🟢'),
        'Clodius Parnassian': ('Least Concern', '🟢'), 'Clouded Sulphur': ('Least Concern', '🟢'),
        'Crescent': ('Least Concern', '🟢'), 'Crimson Patch': ('Least Concern', '🟢'),
        'Eastern Comma': ('Least Concern', '🟢'), 'Great Eggfly': ('Least Concern', '🟢'),
        'Grey Hairstreak': ('Least Concern', '🟢'), 'Indra Swallow': ('Least Concern', '🟢'),
        'Indra Swallowtail': ('Least Concern', '🟢'), 'Julia': ('Least Concern', '🟢'),
        'Large Marble': ('Least Concern', '🟢'), 'Malachite': ('Least Concern', '🟢'),
        'Mangrove Skipper': ('Least Concern', '🟢'), 'Morning Cloak': ('Least Concern', '🟢'),
        'Mourning Cloak': ('Least Concern', '🟢'), 'Orange Oakleaf': ('Least Concern', '🟢'),
        'Orange Tip': ('Least Concern', '🟢'), 'Orchard Swallowtail': ('Least Concern', '🟢'),
        'Painted Lady': ('Least Concern', '🟢'), 'Paper Kite': ('Least Concern', '🟢'),
        'Peacock': ('Least Concern', '🟢'), 'Pine White': ('Least Concern', '🟢'),
        'Pipevine Swallowtail': ('Least Concern', '🟢'), 'Purple Hairstreak': ('Least Concern', '🟢'),
        'Question Mark': ('Least Concern', '🟢'), 'Red Admiral': ('Least Concern', '🟢'),
        'Red Spotted Purple': ('Least Concern', '🟢'), 'Scarce Swallowtail': ('Least Concern', '🟢'),
        'Silver Spot Skipper': ('Least Concern', '🟢'), 'Sixspot Burnet': ('Least Concern', '🟢'),
        'Sootywing': ('Least Concern', '🟢'), 'Southern Dogface': ('Least Concern', '🟢'),
        'Striated Queen': ('Least Concern', '🟢'), 'Two Barred Flasher': ('Least Concern', '🟢'),
        'Ulyses': ('Least Concern', '🟢'), 'Viceroy': ('Least Concern', '🟢'),
        'Wood Satyr': ('Least Concern', '🟢'), 'Yellow Swallowtail': ('Least Concern', '🟢'),
        'Zebra Long Wing': ('Least Concern', '🟢'), 'Copper Tail': ('Not Available', '❓'),
        'Gold Banded': ('Not Available', '❓'), 'Metalmark': ('Not Available', '❓'),
        'Skipper': ('Not Available', '❓'),
    }
    LOCATION_DATA = {
        'American Snout': [29.7604, -95.3698], 'Beckers White': [39.8283, -98.5795],
        'Chestnut': [35.5951, -82.5515], 'Clodius Parnassian': [46.8797, -113.9940],
        'Clouded Sulphur': [41.2033, -77.1945], 'Crescent': [38.6270, -90.1994],
        'Crimson Patch': [29.4241, -98.4936], 'Eastern Comma': [42.3601, -71.0589],
        'Grey Hairstreak': [34.0522, -118.2437], 'Indra Swallowtail': [39.1130, -105.3588],
        'Indra Swallow': [39.1130, -105.3588], 'Julia': [25.7617, -80.1918],
        'Mangrove Skipper': [25.1018, -80.5287], 'Metalmark': [19.4326, -99.1332],
        'Monarch': [37.0902, -95.7129], 'Morning Cloak': [53.9333, -116.5765],
        'Mourning Cloak': [53.9333, -116.5765], 'Pine White': [44.0582, -121.3153],
        'Pipevine Swallowtail': [36.1627, -86.7816], 'Question Mark': [39.9612, -82.9988],
        'Red Spotted Purple': [44.9778, -93.2650], 'Silver Spot Skipper': [40.7128, -74.0060],
        'Skipper': [39.8283, -98.5795], 'Sootywing': [37.7749, -122.4194],
        'Southern Dogface': [32.7767, -96.7970], 'Viceroy': [41.8781, -87.6298],
        'Wood Satyr': [43.6532, -79.3832], 'Yellow Swallowtail': [45.4215, -75.6972],
        'Zebra Long Wing': [27.9944, -81.7603], 'An 88': [-12.0464, -77.0428],
        'Gold Banded': [-23.5505, -46.6333], 'Malachite': [19.2465, -99.1013],
        'Striated Queen': [-15.8267, -47.9218], 'Two Barred Flasher': [-3.7172, -38.5433],
        'Adonis': [48.8566, 2.3522], 'Adonis Blue': [46.8182, 8.2275],
        'Banded Peacock': [28.6139, 77.2090], 'Black Hairstreak': [52.4862, -1.8904],
        'Cabbage White': [51.5074, -0.1278], 'Copper Tail': [35.6895, 139.6917],
        'Large Marble': [36.7783, -119.4179], 'Orange Oakleaf': [13.0827, 80.2707],
        'Orange Tip': [52.5200, 13.4050], 'Painted Lady': [31.7683, 35.2137],
        'Peacock': [55.7558, 37.6173], 'Purple Hairstreak': [53.4808, -2.2426],
        'Red Admiral': [53.3498, -6.2603], 'Scarce Swallowtail': [41.9028, 12.4964],
        'Sixspot Burnet': [47.3769, 8.5417], 'Great Eggfly': [-27.4698, 153.0251],
        'Orchard Swallowtail': [-33.8688, 151.2093], 'Paper Kite': [14.5995, 120.9842],
        'Ulyses': [-16.9203, 145.7710],
    }
    return model, id_to_species, CONSERVATION_STATUS, LOCATION_DATA

model, id_to_species, CONSERVATION_STATUS, LOCATION_DATA = load_resources()

# ==============================================================================
# HELPER FUNCTIONS (Moved to the top)
# ==============================================================================
def get_wiki_info(species_name):
    wiki = wikipediaapi.Wikipedia('BioScanAI/1.0 (your.email@example.com)', 'en')
    page = wiki.page(species_name + " butterfly")
    if not page.exists():
        page = wiki.page(species_name)
    if page.exists():
        return page.summary.split('\n')[0], page.fullurl
    return "No Wikipedia article found.", None

def create_map(species_name):
    if species_name in LOCATION_DATA:
        location = LOCATION_DATA[species_name]
        m = folium.Map(location=location, zoom_start=4, tiles='CartoDB dark_matter')
        folium.Marker(location, popup=f"Habitat of {species_name}", icon=folium.Icon(color='cyan', icon='info-sign')).add_to(m)
        return m
    return None

def predict_butterfly(img):
    img_array = image.img_to_array(img)
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    species_pred, attribute_pred = predictions[0], predictions[1]
    species_id = np.argmax(species_pred[0])
    species_name = id_to_species[species_id]
    confidence = np.max(species_pred[0]) * 100
    is_mimic_prob = attribute_pred[0][0]
    is_model_prob = attribute_pred[0][1]
    return species_name, confidence, is_mimic_prob, is_model_prob

# ==============================================================================
# STREAMLIT APP LAYOUT
# ==============================================================================

st.markdown("<h1>🦋 BioScan AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Your AI-Powered Butterfly Expert System</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload an image to begin analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        with st.spinner('AI is analyzing the butterfly...'):
            species, conf, mimic_prob, model_prob = predict_butterfly(image.load_img(uploaded_file))
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔍 Analysis Report")
        
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            st.metric(label="Predicted Species", value=species)
        with pred_col2:
            st.metric(label="Confidence", value=f"{conf:.2f}%")
        
        if mimic_prob > 0.5: st.info(f"Ecological Trait: This is likely a MIMIC species.")
        if model_prob > 0.5: st.warning(f"Ecological Trait: This is likely a MODEL species (toxic).")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📚 Species Encyclopedia")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("<h4>Conservation & Facts</h4>", unsafe_allow_html=True)
        status, emoji = CONSERVATION_STATUS.get(species, ("Not Available", "❓"))
        st.write(f"**Status:** {status} {emoji}")
        summary, url = get_wiki_info(species)
        st.write(f"**Wikipedia:** {summary}")
        if url: st.markdown(f"[Read more on Wikipedia]({url})")

    with info_col2:
        st.markdown("<h4>Geographic Habitat</h4>", unsafe_allow_html=True)
        # This is the line that caused the error. Now it will work.
        butterfly_map = create_map(species)
        if butterfly_map: st_folium(butterfly_map, width=700, height=350)
        else: st.write("Habitat map not available.")
    st.markdown("</div>", unsafe_allow_html=True) 