# 🚀 Quick Start - GitHub Actions + Railway Setup

Potpuna automatizacija u **5 koraka**!

---

## ✅ Trenutno stanje

- ✅ Projekat radi lokalno
- ✅ 20 članaka sa sentiment/topics/summaries
- ✅ Streamlit dashboard na http://localhost:8501
- ✅ GitHub Actions workflow fajl spreman
- ✅ Railway konfiguracija spremna

---

## 🎯 Cilj

1. **GitHub Actions** → Automatski fetch novih članaka svaki dan
2. **Railway** → 24/7 javno dostupan dashboard

---

## 📝 CHECKLIST

### STEP 1: Push na GitHub (5 minuta)

```bash
cd /home/David/Documents/GithubRepos/news-trend-analysis

# Proveri current branch
git branch

# Dodaj sve fajlove
git add .

# Commit
git commit -m "🚀 Setup automatic updates & Railway deployment"

# Ako nemaš remote, dodaj:
git remote add origin https://github.com/YOUR_USERNAME/news-trend-analysis.git

# Push (zameni 'main' sa svojim branch-om ako je drugačiji)
git push -u origin main
```

**Ako nemaš GitHub repo:**
1. Idi na https://github.com/new
2. Repository name: `news-trend-analysis`
3. Public (za besplatne GitHub Actions minutes)
4. Don't add README/gitignore (već imaš)
5. Create repository
6. Koristi command-e iznad sa svojim URL-om

✅ **Provera:** Repo je na GitHub-u sa svim fajlovima

---

### STEP 2: GitHub Secret - API Key (2 minuta)

1. **Idi na:**
   ```
   https://github.com/YOUR_USERNAME/news-trend-analysis/settings/secrets/actions
   ```

2. **Klikni:** "New repository secret"

3. **Unesi:**
   - Name: `NEWS_API_KEY`
   - Secret: `YOUR_NEWSDATA_IO_API_KEY`

4. **Klikni:** "Add secret"

✅ **Provera:** Secret postoji u listi

---

### STEP 3: Enable GitHub Actions (1 minut)

1. **Idi na:**
   ```
   https://github.com/YOUR_USERNAME/news-trend-analysis/settings/actions
   ```

2. **Scroll do "Workflow permissions"**

3. **Select:** "Read and write permissions"

4. **Check:** ✅ "Allow GitHub Actions to create and approve pull requests"

5. **Save**

6. **Idi na Actions tab:**
   ```
   https://github.com/YOUR_USERNAME/news-trend-analysis/actions
   ```

7. **Ako vidiš banner "Workflows disabled":**
   - Klikni "I understand my workflows, go ahead and enable them"

✅ **Provera:** Actions tab prikazuje "Daily News Update" workflow

---

### STEP 4: Test GitHub Actions (3 minuta)

1. **U Actions tab-u:**
   ```
   https://github.com/YOUR_USERNAME/news-trend-analysis/actions
   ```

2. **Klikni na:** "Daily News Update"

3. **Klikni:** "Run workflow" ▶️
   - Branch: `main`
   - Klikni: "Run workflow"

4. **Prati izvršavanje:**
   - Zeleni krug = running
   - Zeleni checkmark = success ✅
   - Crveni X = failed ❌

5. **Proveri commit:**
   - Novi commit: `🤖 Daily update: 2025-10-30...`
   - Fajlovi ažurirani u `data/processed/`

✅ **Provera:** Workflow uspešno završio, novi commit na repo-u

---

### STEP 5: Railway Deployment (10 minuta)

1. **Idi na:** https://railway.app

2. **Login:** "Login with GitHub" (preporučeno)

3. **New Project:**
   - Klikni: "New Project"
   - Select: "Deploy from GitHub repo"
   - Autorizuj Railway (prvi put)
   - Izaberi: `news-trend-analysis`

4. **Sačekaj build:** (~5-10 min prvi put)

5. **Dodaj Environment Variables:**
   - Klikni na service → "Variables" tab
   - Add variable:
     ```
     NEWS_API_KEY=YOUR_NEWSDATA_IO_API_KEY
     ```
   - Save (automatski restart)

6. **Generate Domain:**
   - Settings tab → Networking
   - Klikni: "Generate Domain"
   - Dobijanje URL-a: `https://xxx.up.railway.app`

7. **Open URL:**
   - Dashboard je live! 🎉

8. **Optional - Sleep on inactivity:**
   - Settings → Service
   - Toggle: "Sleep after inactivity"
   - (Štedi free hours)

✅ **Provera:** Dashboard dostupan na public URL-u

---

## 🎉 Gotovo!

### Šta sada radi automatski:

**Svaki dan u 8:00 UTC (9:00 CET / 10:00 CEST):**
1. ⏰ GitHub Actions se pokreće
2. 📰 Fetch-uje 50 novih članaka
3. 🕷️ Scrape-uje full content
4. 😊 Sentiment analiza
5. 📊 Topic modeling (automatski nazivi)
6. 📝 Summaries
7. 💾 Commit + push na GitHub
8. 🚂 Railway detektuje → Auto redeploy
9. ✅ Dashboard ima fresh podatke!

---

## 📊 URLs

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

**Public Dashboard:**
```
https://news-trend-analysis-production.up.railway.app
```
(Ili šta god je tvoj generated domain)

---

## 🔄 Maintenance

### Ručno pokretanje update-a:
```bash
# GitHub Actions
# → Actions tab → Run workflow

# Lokalno (za test)
python daily_update.py --max-results 10
git add data/ models/
git commit -m "Manual update"
git push
```

### Promena schedule-a:
```bash
# Otvori .github/workflows/daily-update.yml
# Promeni cron: '0 8 * * *'
# Commit + push
```

### Monitoring:
- **GitHub Actions:** Vidi logove i status
- **Railway:** Vidi deployment status i app logs

---

## 📚 Detaljne instrukcije

- **GitHub Actions:** Vidi `.github/GITHUB_ACTIONS_SETUP.md`
- **Railway:** Vidi `RAILWAY_SETUP.md`
- **Lokalna automatizacija:** Vidi `AUTOMATION.md`

---

## 🐛 Troubleshooting

### GitHub Actions ne radi:
1. Proveri da li je secret `NEWS_API_KEY` dodat
2. Proveri workflow permissions (Read and write)
3. Proveri logove u Actions tab

### Railway deployment fails:
1. Proveri `requirements.txt` (sve dependencies)
2. Proveri Railway logs za specific error
3. Verifikuj da `Procfile` postoji

### Dashboard prikazuje stare podatke:
1. GitHub Actions treba da commit-uje nove podatke
2. Railway treba da detektuje commit i redeploy-uje
3. Možda trebaš manual push postojećih podataka

---

## ✨ Sledeći koraci

1. ✅ **Share dashboard URL** sa prijateljima/kolegama
2. 🎨 **Customize Streamlit app** (boje, layout)
3. 📈 **Add more features** (email alerts, export PDF)
4. 🌐 **Custom domain** (ako imaš svoj)
5. 💰 **Monitor free tier limits** (upgrade ako treba)

---

**Sve je spremno! Uživaj u automatizovanom news analysis dashboardu! 🎉**
