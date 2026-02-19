# 💨 Digital Wind Tunnel Pro

**Digital Wind Tunnel Pro** is a web-based aerodynamic analysis tool built with Python and Streamlit. It allows students, engineers, and hobbyists to simulate, analyze, and compare the performance of different airfoils in a virtual environment.

Powered by [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) and [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox), this tool provides rapid aerodynamic predictions without the need for computationally expensive CFD software.

## Features

* **Airfoil Generation & Import**:
    * **NACA 4-Digit Series**: Generate standard airfoils (e.g., 2412, 0012).
    * **NACA 5-Digit Series**: Support for complex camber lines (e.g., 23012).
    * **NACA 6-Series**: Advanced support with auto-fetch capabilities from online databases (e.g., 63-415).
    * **Custom Upload**: Import your own `.dat` or `.txt` coordinate files.
* **Comparison Mode**: Simultaneously analyze and overlay two different airfoils to compare performance directly.
* **Physics-Informed Analysis**:
    * Adjustable Reynolds Number ($Re$) and Model Size.
    * **Single Point Analysis**: Quick check for Lift ($C_L$), Drag ($C_D$), and Moment ($C_M$) at a specific Angle of Attack ($\alpha$).
    * **Polar Sweep**: Full alpha sweep simulation (-30° to +30°) to generate Lift Curves, Drag Polars, and L/D ratios.
* **Interactive Visualizations**:
    * Real-time Airfoil Geometry plotting.
    * Interactive charts for $C_L$ vs $\alpha$, $C_L$ vs $C_D$, and Efficiency ($L/D$).
* **Data Export**: Download your simulation results as a CSV file for further analysis in Excel or MATLAB.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/digitalwindtunnel.git](https://github.com/your-username/digitalwindtunnel.git)
    cd digitalwindtunnel
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
