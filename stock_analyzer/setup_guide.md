# 📊 Stock Financial Dashboard - Complete Setup Guide

## 🚀 Quick Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Create Project Directory
```bash
mkdir stock_dashboard
cd stock_dashboard
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv dashboard_env

# Activate virtual environment
# On Windows:
dashboard_env\Scripts\activate
# On macOS/Linux:
source dashboard_env/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Create Project Files
Save the following files in your project directory:
- `main.py` - Main application
- `data_fetcher.py` - Data fetching module
- `technical_analysis.py` - Technical analysis module
- `financial_charts.py` - Financial charts module
- `analytics.py` - Analytics and calculations module
- `ui_components.py` - UI components module
- `requirements.txt` - Dependencies file

### 5. Run the Dashboard
```bash
streamlit run main.py
```

## 📋 Core Dependencies Explained

### **Essential Libraries:**
- **streamlit** (>=1.28.0) - Web framework for the dashboard interface
- **pandas** (>=2.0.0) - Data manipulation and analysis
- **numpy** (>=1.24.0) - Numerical computing
- **yfinance** (>=0.2.18) - Yahoo Finance API for stock data
- **plotly** (>=5.15.0) - Interactive charts and visualizations

### **Optional Performance Enhancements:**
- **numba** - JIT compilation for faster numerical calculations
- **pyarrow** - Faster data serialization and file I/O

### **Optional Analysis Extensions:**
- **scipy** - Advanced statistical functions
- **scikit-learn** - Machine learning capabilities
- **statsmodels** - Statistical modeling

## 🔧 Alternative Installation Methods

### Method 1: Conda Environment
```bash
# Create conda environment
conda create -n dashboard python=3.10
conda activate dashboard

# Install packages
conda install streamlit pandas numpy plotly
pip install yfinance
```

### Method 2: Docker Setup
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📱 Platform-Specific Notes

### **Windows:**
- Ensure Python is added to PATH
- Use Command Prompt or PowerShell
- Virtual environment activation: `dashboard_env\Scripts\activate`

### **macOS:**
- Install via Homebrew: `brew install python`
- Virtual environment activation: `source dashboard_env/bin/activate`

### **Linux:**
- Install python3-pip: `sudo apt install python3-pip`
- Virtual environment activation: `source dashboard_env/bin/activate`

## 🚨 Troubleshooting

### Common Issues:

**1. Import Errors:**
```bash
# Update pip
pip install --upgrade pip

# Reinstall problematic package
pip uninstall yfinance
pip install yfinance
```

**2. Streamlit Port Issues:**
```bash
# Run on different port
streamlit run main.py --server.port 8502
```

**3. Data Fetching Errors:**
- Check internet connection
- Yahoo Finance may have rate limits
- Try different stock tickers

**4. Memory Issues with Large Datasets:**
```bash
# Install performance packages
pip install numba pyarrow
```

## 🔒 Security & Performance

### **API Rate Limits:**
- Yahoo Finance: ~2000 requests/hour
- Built-in caching reduces API calls
- Data cached for 5-60 minutes

### **Performance Optimization:**
- Use virtual environment
- Install optional performance packages
- Clear browser cache if charts don't load

### **Data Privacy:**
- No personal data stored
- All data fetched from public APIs
- Session data cleared on browser close

## 🌐 Deployment Options

### **Local Development:**
```bash
streamlit run main.py
# Access at: http://localhost:8501
```

### **Streamlit Cloud:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### **Heroku:**
```bash
# Create Procfile
echo "web: streamlit run main.py --server.port $PORT --server.enableCORS false" > Procfile

# Deploy
git init
heroku create your-dashboard-name
git push heroku main
```

## 📊 Feature Requirements

### **Minimum System Requirements:**
- RAM: 2GB minimum, 4GB recommended
- Storage: 100MB for application + cache
- Network: Stable internet for data fetching
- Browser: Chrome, Firefox, Safari, Edge (latest versions)

### **Recommended Setup:**
- RAM: 8GB+ for large datasets
- SSD storage for faster caching
- High-speed internet for real-time data

## 🆕 Version Compatibility

### **Python Versions:**
- ✅ Python 3.8 - 3.11 (Recommended)
- ⚠️ Python 3.7 (Limited support)
- ❌ Python 3.6 and below (Not supported)

### **Browser Compatibility:**
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🔄 Update Instructions

### **Update All Packages:**
```bash
pip install --upgrade -r requirements.txt
```

### **Update Specific Package:**
```bash
pip install --upgrade streamlit
```

### **Check Versions:**
```bash
pip list | grep streamlit
streamlit version
```

## 📞 Support

### **Getting Help:**
- Check GitHub issues for common problems
- Streamlit documentation: https://docs.streamlit.io
- yfinance documentation: https://pypi.org/project/yfinance/

### **Contributing:**
- Report bugs via GitHub issues
- Submit feature requests
- Contribute code improvements

---

**🎉 You're all set! Run `streamlit run main.py` and start analyzing stocks!**