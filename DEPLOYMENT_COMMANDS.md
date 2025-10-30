# 🎯 FINAL DEPLOYMENT COMMANDS

Copy-paste komande za potpuni setup!

---

## 📋 PRE-FLIGHT CHECKLIST

✅ Projekat radi lokalno  
✅ `data/processed/articles_with_summary.csv` postoji  
✅ Imaš GitHub nalog  
✅ Imaš Railway nalog (ili ćeš napraviti)  
✅ API key: `YOUR_NEWSDATA_IO_API_KEY`

---

## 🚀 DEPLOYMENT U 5 KORAKA

### STEP 1: Git Commit & Push (5 min)

```bash
cd /home/David/Documents/GithubRepos/news-trend-analysis

# Check status
git status

# Add all files
git add .

# Commit
git commit -m "🚀 Setup automatic updates & Railway deployment"

# If no remote exists, add it (REPLACE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/news-trend-analysis.git

# Push
git push -u origin main

# If 'main' doesn't exist, try:
# git branch -M main
# git push -u origin main
```

**PAUSE:** Verifikuj da je kod na GitHub-u!

---

### STEP 2: GitHub Secret (2 min)

**Browser:**
1. Open: `https://github.com/YOUR_USERNAME/news-trend-analysis/settings/secrets/actions`
2. Click: **"New repository secret"**
3. Name: `NEWS_API_KEY`
4. Secret: `YOUR_NEWSDATA_IO_API_KEY`
5. Click: **"Add secret"**

**PAUSE:** Verifikuj da secret postoji u listi!

---

### STEP 3: Enable GitHub Actions (2 min)

**Browser:**
1. Open: `https://github.com/YOUR_USERNAME/news-trend-analysis/settings/actions`
2. Scroll to **"Workflow permissions"**
3. Select: **"Read and write permissions"**
4. Check: ✅ **"Allow GitHub Actions to create and approve pull requests"**
5. Click: **"Save"**

**PAUSE:** Verifikuj permissions!

---

### STEP 4: Test GitHub Actions (3 min)

**Browser:**
1. Open: `https://github.com/YOUR_USERNAME/news-trend-analysis/actions`
2. If see "Workflows disabled" banner → Click **"Enable"**
3. Click on: **"Daily News Update"** workflow
4. Click: **"Run workflow"** ▶️
   - Branch: `main`
   - Click: **"Run workflow"**
5. Wait ~2-3 minutes
6. Should see: ✅ Green checkmark

**PAUSE:** Verifikuj da je workflow uspeo! Check commits.

---

### STEP 5: Railway Deployment (10 min)

**Browser:**

**5.1 Create Project**
1. Open: https://railway.app
2. Click: **"Login with GitHub"** (or create account)
3. Click: **"New Project"**
4. Select: **"Deploy from GitHub repo"**
5. Authorize Railway (first time)
6. Select repo: **`news-trend-analysis`**
7. Wait for build (~5-10 min)

**5.2 Add Environment Variable**
1. Click on your service (deployment)
2. Tab: **"Variables"**
3. Click: **"+ New Variable"**
4. Add:
   ```
   NEWS_API_KEY=YOUR_NEWSDATA_IO_API_KEY
   ```
5. Service will auto-restart

**5.3 Generate Public Domain**
1. Tab: **"Settings"**
2. Section: **"Networking"**
3. Click: **"Generate Domain"**
4. Copy URL: `https://news-trend-analysis-production-XXXX.up.railway.app`

**5.4 Test Dashboard**
1. Open generated URL
2. Should see Streamlit dashboard with your data! 🎉

**5.5 Optional: Sleep on Inactivity**
1. Settings → Service
2. Toggle ON: **"Sleep after inactivity"**
3. Saves free tier hours (app wakes in 2-3s)

---

## ✅ VERIFICATION

### GitHub Actions:
```bash
# Check in browser:
https://github.com/YOUR_USERNAME/news-trend-analysis/actions

# Should see:
✅ "Daily News Update" workflow
✅ Scheduled for 8:00 UTC daily
✅ Can run manually via "Run workflow"
```

### Railway:
```bash
# Check in browser:
https://railway.app/dashboard

# Should see:
✅ Service running
✅ Public domain generated
✅ Environment variables set
✅ Build successful
```

### Dashboard:
```bash
# Open your Railway URL
https://news-trend-analysis-production-XXXX.up.railway.app

# Should see:
✅ Streamlit dashboard loads
✅ Articles displayed
✅ Sentiment analysis visible
✅ Topics with automatic labels
✅ Summaries shown
```

---

## 🎉 SUCCESS!

Sada imaš:

✅ **GitHub Actions** → Automatski fetch novih članaka svaki dan u 8:00 UTC  
✅ **Railway** → 24/7 javno dostupan dashboard  
✅ **Auto-deploy** → Svaki push na GitHub → Railway auto-deploy  
✅ **Free hosting** → Besplatno (sa limitima)

---

## 📊 URLS - SAVE THESE!

**GitHub Repository:**
```
https://github.com/YOUR_USERNAME/news-trend-analysis
```

**GitHub Actions:**
```
https://github.com/YOUR_USERNAME/news-trend-analysis/actions
```

**Railway Dashboard:**
```
https://railway.app/dashboard
```

**Public Dashboard (SHARE THIS!):**
```
https://news-trend-analysis-production-XXXX.up.railway.app
```

---

## 🔄 DAILY WORKFLOW

**Automatski (bez tvog inputa):**

```
8:00 UTC (9:00 CET / 10:00 CEST)
│
├─ GitHub Actions se pokreće
│  ├─ Fetch 50 novih članaka
│  ├─ Scrape full content
│  ├─ Sentiment + Topics + Summaries
│  └─ Commit + push na GitHub
│
├─ Railway detektuje novi commit
│  ├─ Automatski re-deploy (~2 min)
│  └─ Dashboard ima fresh podatke!
│
└─ ✅ Sve automatski! Zero manual work!
```

---

## 🛠️ MAINTENANCE COMMANDS

### Ručno pokretanje update-a:

**GitHub Actions (Browser):**
```
1. Go to: https://github.com/YOUR_USERNAME/news-trend-analysis/actions
2. Click: "Daily News Update"
3. Click: "Run workflow" ▶️
```

**Lokalno (Terminal):**
```bash
cd /home/David/Documents/GithubRepos/news-trend-analysis
source .venv/bin/activate
python daily_update.py --max-results 50

# Push ako želiš da deploy-uješ
git add data/ models/
git commit -m "Manual update"
git push
```

### Promena schedule-a:

```bash
# Otvori .github/workflows/daily-update.yml
nano .github/workflows/daily-update.yml

# Promeni:
# cron: '0 8 * * *'  # 8:00 UTC
# U:
# cron: '0 */6 * * *'  # Svaka 6 sati

git add .github/workflows/daily-update.yml
git commit -m "Update schedule to every 6 hours"
git push
```

### Provera logova:

**GitHub Actions:**
```
https://github.com/YOUR_USERNAME/news-trend-analysis/actions
→ Click na run → Expand steps
```

**Railway:**
```
https://railway.app/dashboard
→ Click service → "Deployments" tab → Click deployment → Logs
```

---

## 🐛 QUICK TROUBLESHOOTING

### GitHub Actions failed?
```bash
1. Check: https://github.com/YOUR_USERNAME/news-trend-analysis/actions
2. Click failed run → See error
3. Common fixes:
   - Secret not added → Add NEWS_API_KEY
   - Permissions → Settings → Actions → Read/Write
   - Rate limit → Wait 24h (API limit)
```

### Railway deployment failed?
```bash
1. Check: Railway Dashboard → Logs
2. Common fixes:
   - Missing requirements → Check requirements.txt
   - Build timeout → Try redeploy
   - Environment variable → Add NEWS_API_KEY
```

### Dashboard shows no data?
```bash
# Run locally to generate data:
python daily_update.py --max-results 10

# Push to GitHub:
git add data/ models/
git commit -m "Add initial data"
git push

# Railway will auto-deploy!
```

---

## 📚 FULL DOCUMENTATION

- **[QUICK_START.md](QUICK_START.md)** - Detaljne instrukcije
- **[AUTOMATION.md](AUTOMATION.md)** - Sve opcije automatizacije
- **[RAILWAY_SETUP.md](RAILWAY_SETUP.md)** - Railway deployment detalji
- **[.github/GITHUB_ACTIONS_SETUP.md](.github/GITHUB_ACTIONS_SETUP.md)** - GitHub Actions guide

---

## 🎊 GOTOVO!

**Čestitam! Tvoj News Trend Analysis projekat je sada:**

✅ Potpuno automatizovan  
✅ Javno dostupan 24/7  
✅ Self-updating svaki dan  
✅ Production-ready!

**Share your dashboard URL sa svetom! 🌍✨**

---

**Need help?** Pogledaj documentation fajlove gore! 📖
