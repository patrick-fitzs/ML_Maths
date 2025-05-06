# 📘 Basic Maths & Statistics for Machine Learning (Beginner-Friendly)

This project introduces fundamental mathematical and statistical concepts commonly used in Machine Learning, with clear Python examples and visualisations. It's perfect for beginners looking to build a strong foundation before diving into ML models.

---

## 📂 Files Included

### `BasicMathsForMl.py` — *Basic Statistics*
Introduces core statistical tools used in data analysis and preprocessing:
- **Mean** – average value
- **Median** – middle value in a sorted list
- **Mode** – most frequent value (`scipy.stats.mode`)
- **Standard Deviation** – how spread out values are
- **Variance** – square of standard deviation
- **Percentiles** – useful for understanding data distribution

---

### `BiggerData.py` — *Larger Datasets & Distributions*
Demonstrates working with larger, randomly generated datasets:
- **Uniform distribution** using `np.random.uniform`
- **Histograms** for visualising distribution
- **Normal distribution** using `np.random.normal`
- **Scatter plots** to explore relationships between variables

---

### `Regression.py` — *Linear and Polynomial Regression*
Introduces regression models to explore relationships and make predictions:

#### 🔹 Linear Regression
- Uses `scipy.stats.linregress` to fit a line
- Outputs:
  - **Slope** and **intercept**
  - **Correlation coefficient (`r`)**
  - **P-value** and **standard error**
- Visualises the line of best fit over real data

#### 🔹 Polynomial Regression
- Fits curves using `np.polyfit` and `np.poly1d`
- Measures fit using **R² score** (`sklearn.metrics.r2_score`)
- Great for when data doesn't follow a straight line pattern

---