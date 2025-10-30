# 🔑 API Key Setup Guide

## Where to Put Your NewsAPI Key Locally

You have **3 easy options** to set your API key:

---

## ✅ **Option 1: .env File (RECOMMENDED)** 

This is the **easiest and most secure** method!

### Steps:

1. **Get Your API Key**
   - Go to: https://newsapi.org/
   - Sign up for a free account
   - Copy your API key

2. **Edit the .env File**
   ```bash
   # Open the .env file in VS Code
   code .env
   ```
   
3. **Replace the placeholder with your actual key:**
   ```bash
   # Before:
   NEWS_API_KEY=your_api_key_here
   
   # After (example):
   NEWS_API_KEY=1234567890abcdef1234567890abcdef
   ```

4. **Save the file** (Ctrl+S)

5. **Run the script:**
   ```bash
   python src/fetch_news.py
   ```

### ✅ **Advantages:**
- ✅ Secure (`.env` is already in `.gitignore`)
- ✅ Permanent (works every time)
- ✅ No need to remember to export
- ✅ Works with all scripts automatically

---

## **Option 2: Environment Variable (Temporary)**

Set the key in your current terminal session:

```bash
export NEWS_API_KEY='your_actual_api_key_here'
```

Then run:
```bash
python src/fetch_news.py
```

### ⚠️ Note:
- Only works in the current terminal session
- Needs to be set again if you close the terminal

---

## **Option 3: Shell Profile (Permanent)**

Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
# Open your bash profile
nano ~/.bashrc

# Add this line at the end:
export NEWS_API_KEY='your_actual_api_key_here'

# Save and reload
source ~/.bashrc
```

---

## 🎯 **Which Method Should I Use?**

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **.env file** | **Most people** | ✅ Easy, Secure, Permanent | Need python-dotenv |
| **export command** | Quick testing | Fast | Temporary |
| **Shell profile** | System-wide | Always available | Less secure, affects all projects |

### **Recommendation:** Use the **.env file** method! 🏆

---

## 📝 Current Setup

Your project is already configured to use `.env` files!

**Files Created:**
- ✅ `.env` - Your local configuration (add your key here)
- ✅ `.env.example` - Template for others
- ✅ `.gitignore` - Protects your `.env` from being committed

**Code Updated:**
- ✅ `src/fetch_news.py` - Now reads from `.env` automatically
- ✅ `requirements.txt` - Added `python-dotenv`
- ✅ Package installed - Ready to use

---

## 🚀 Quick Start

**Just 3 steps:**

1. **Get your key:** https://newsapi.org/
2. **Edit `.env`:** Replace `your_api_key_here` with your actual key
3. **Run:** `python src/fetch_news.py`

That's it! 🎉

---

## 🔒 Security Notes

✅ **Safe:**
- Your `.env` file is in `.gitignore`
- It won't be committed to Git
- It's safe to store your key there

❌ **Don't:**
- Don't commit `.env` to Git
- Don't share your API key publicly
- Don't hardcode keys in your Python files

---

## 🧪 Test Your Setup

Run this to check if your key is loaded:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key found!' if os.getenv('NEWS_API_KEY') != 'your_api_key_here' else 'Please add your API key to .env')"
```

Or just run:
```bash
python src/fetch_news.py
```

If configured correctly, it will fetch news articles!

---

## 🐛 Troubleshooting

### "No API key found" error?
1. Check that `.env` file exists
2. Make sure you replaced `your_api_key_here` with your actual key
3. No quotes needed around the key in `.env`
4. Save the file after editing

### "API Error: Unauthorized" ?
- Your API key might be invalid
- Check it at: https://newsapi.org/account
- Make sure you copied it correctly

### Still not working?
Use the export method temporarily:
```bash
export NEWS_API_KEY='your_key_here'
python src/fetch_news.py
```

---

## 📍 File Locations

```
news-trend-analysis/
├── .env                  ← ADD YOUR KEY HERE!
├── .env.example          ← Template/example
├── .gitignore           ← Protects .env
└── src/
    └── fetch_news.py    ← Reads from .env automatically
```

---

## ✅ Summary

**To add your API key locally:**

1. Open `.env` file
2. Replace `your_api_key_here` with your actual NewsAPI key
3. Save the file
4. Run `python src/fetch_news.py`

**That's all!** 🎊

Your key is now secure, permanent, and ready to use!
