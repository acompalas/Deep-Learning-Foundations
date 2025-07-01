import streamlit as st
import streamlit.components.v1 as components
import base64
import os

# -- Encode local image as base64 data URI --
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

# -- Render a single slide from data --
def render_slide(slide: dict, active=False) -> str:
    class_name = "slide active" if active else "slide"
    return f"""
    <div class="{class_name}">
        <img src="{slide['image']}" alt="Slide Image">
        <div class="overlay">
            <h2>{slide['title']}</h2>
            <p>{slide['description']}</p>
            <div class="slide-buttons">
                {''.join(f'<a href="?page={btn["href"].lower()}" class="learn-more">{btn["label"]}</a>' for btn in slide["buttons"])}
            </div>
        </div>
    </div>
    """

# -- Main loader for component --
def load_hero_component(slides, height=400):
    component_dir = os.path.join("components", "test_component")

    with open(os.path.join(component_dir, "style.css"), "r", encoding="utf-8") as f:
        css = f"<style>{f.read()}</style>"

    with open(os.path.join(component_dir, "script.js"), "r", encoding="utf-8") as f:
        js = f"<script>{f.read()}</script>"

    slide_html = "\n".join(render_slide(slide, active=(i == 0)) for i, slide in enumerate(slides))

    html = f"""
    <div class="carousel-container">
        {slide_html}
        <div class="nav-buttons">
            <button class="prev">&#9664;</button>
            <button class="next">&#9654;</button>
        </div>
    </div>
    """

    full_html = f"{css}\n{html}\n{js}"
    components.html(full_html, height=height, scrolling=False)

# -- Redirect to actual page file if ?page=... is present
PAGE_BASE_NAMES = [
    "ordinary_least_squares",
    "linear_regression",
    "logistic_regression",
    "regularization",
    "perceptron",
    "mlp",
    "optimizers"
]

# ✅ This catches both str and list values
query_page = st.query_params.get("page")
if isinstance(query_page, list):
    query_page = query_page[0]
if query_page:
    query_page = query_page.lower()
    if query_page in PAGE_BASE_NAMES:
        index = PAGE_BASE_NAMES.index(query_page)
        filename = f"{index+1:02d}_{query_page}.py"
        print(f"[DEBUG] Switching to pages/{filename}", file=sys.stderr)  # Print to terminal
        st.switch_page(f"pages/{filename}")
    else:
        print(f"[DEBUG] Unknown page: {query_page}", file=sys.stderr)
        st.error(f"Unknown page: {query_page}")

# -- Main Page --
st.set_page_config(page_title="Hero Slide Demo", layout="wide")
st.title("Test Hero Component")

# Load and encode images
image_dir = "assets/icons"
image_dict = {}

for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        name = os.path.splitext(filename)[0]
        path = os.path.join(image_dir, filename)
        image_dict[name] = encode_image(path)

# Slide definitions
slides = [
    {
        "image": image_dict["ols"],
        "title": "Ordinary Least Squares",
        "description": "Minimize squared errors to fit the best line.",
        "buttons": [
            {"label": "Learn More", "href": "ordinary_least_squares"},
        ]
    },
    {
        "image": image_dict["mlp"],
        "title": "Multi-Layer Perceptron",
        "description": "Stack nonlinear layers to model complex patterns.",
        "buttons": [
            {"label": "View Guide", "href": "mlp"}
        ]
    }
]

# Run the component
load_hero_component(slides)
