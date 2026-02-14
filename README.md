# 1D Resistivity Forward Modeling App

A local web application for modeling 1D DC Resistivity (Schlumberger Array) using Python and Streamlit.

## Features
- **Interactive Layer Modeling**: Edit layer thickness and resistivity in a table.
- **Real-time Visualization**:
    - Layered Earth Model (Step Plot).
    - Apparent Resistivity Sounding Curve (Log-Log).
- **Accurate Calculation**: Uses `empymod` for robust electromagnetic modeling in the DC limit.

## Installation

1.  Clone or download this repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the application:
    ```bash
    streamlit run app.py
    ```
2.  The app will open in your default web browser.
3.  Use the sidebar to adjust survey parameters (AB/2 range).
4.  Edit the table to change the layer model.

## Dependencies
- `streamlit`
- `solara` (not used, just Streamlit)
- `empymod`
- `plotly`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
