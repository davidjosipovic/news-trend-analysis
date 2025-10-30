# 🚂 Railway Deployment Guide

Deploy Streamlit dashboard na Railway za 24/7 uptime!

---

## 📋 Preduslovi

- GitHub repository sa push-ovanim kodom
- Railway nalog (besplatan): https://railway.app
- Postojeći podaci u `data/processed/articles_with_summary.csv`

---

## 🚀 STEP 1: Kreiranje Railway Projekta

### 1. Idi na Railway:
```
https://railway.app
```

### 2. Sign up / Login:
- **Best option:** "Login with GitHub" (automatski sync)
- Alternative: Email/Google

### 3. Novi projekat:
- Klikni: **"New Project"**
- Select: **"Deploy from GitHub repo"**
- Autorizuj Railway za pristup GitHub-u (ako prvi put)

### 4. Izaberi repository:
- Traži: `news-trend-analysis`
- Klikni na repo
- Railway automatski detektuje Python projekat! 🎉

---

## ⚙️ STEP 2: Konfiguracija Environment Variables

Railway automatski build-uje, ali trebaju env variables:

### 1. U Railway Dashboard:
- Klikni na svoj service (deployment)
- Idi na: **"Variables"** tab

### 2. Dodaj varijable:
```bash
PORT=8501
NEWS_API_KEY=YOUR_NEWSDATA_IO_API_KEY
```

### 3. Save changes
Railway će automatski restart-ovati sa novim varijablama!

---

## 🔧 STEP 3: Verifikacija Deploy-a

Railway koristi:
- ✅ `Procfile` → Automatski detektovan
- ✅ `requirements.txt` → Automatski instalira dependencies
- ✅ `runtime.txt` → Python 3.11.9
- ✅ `railway.json` → Build & deploy konfiguracija

### Build time:
- Prvi deploy: ~5-10 minuta (install svih paketa)
- Update-ovi: ~2-3 minuta (cached dependencies)

### Proveri logove:
1. Klikni na deployment
2. Tab: **"Deployments"**
3. Klikni na aktivan deployment
4. Gledaj live build logove

### Kada je gotovo:
```
✅ Build successful
✅ Deployment live
```

---

## 🌐 STEP 4: Dobijanje Public URL-a

### 1. Generate domain:
- U Railway dashboardu → Tvoj service
- Tab: **"Settings"**
- Section: **"Networking"**
- Klikni: **"Generate Domain"**

### 2. Dobit ćeš URL:
```
https://news-trend-analysis-production.up.railway.app
```
(ili sličan)

### 3. Test:
- Klikni na URL
- Streamlit dashboard se otvara! 🎉
- Vidiš sve tvoje članke, topics, summaries

---

## 🔄 STEP 5: Automatski Re-Deploy

Railway je **conectovan sa GitHub-om**, tako da:

✅ **Svaki `git push`** → Automatski re-deploy!  
✅ **GitHub Actions commit-uje nove podatke** → Railway detektuje → Re-deploy  
✅ **Dashboard se automatski ažurira** sa novim člancima!

### Workflow:
```
1. GitHub Actions (8:00 UTC svaki dan)
   ↓ Fetch 50 novih članaka
   ↓ Scrape + Sentiment + Topics + Summaries
   ↓ Git commit + push

2. Railway detektuje novi commit
   ↓ Automatski re-deploy (~2 min)
   ↓ Dashboard ima fresh podatke! ✅
```

---

## 📊 STEP 6: Monitoring

### Railway Dashboard:
```
https://railway.app/dashboard
```

### Proveri:
- **Deployments:** Status svih deploy-ova
- **Metrics:** CPU, RAM, Network usage
- **Logs:** Live aplikacijski logovi
- **Usage:** Troškovi (besplatno do limita)

### Email notifikacije:
Railway šalje email ako deployment fail-uje!

---

## 💰 Railway Pricing (Free Tier)

**Hobby Plan (Besplatno):**
- ✅ $5 credit mesečno
- ✅ 500h runtime/mesec
- ✅ 100GB outbound bandwidth
- ✅ Public GitHub repos
- ❌ Private repos (treba upgrade)

**Za ovaj projekat:**
- Streamlit dashboard: ~1-2 MB RAM
- Continuous run: 24/7 = ~720h/mesec
- **Problem:** Prelazi 500h limit! 😱

### Rešenje: Sleep after inactivity

Railway može staviti app u sleep mode posle 5min neaktivnosti:

1. Settings → Service
2. Toggle: **"Sleep after inactivity"**
3. App se budi automatski pri prvom requestu (2-3s)

**Rezultat:**
- App aktivan samo kad ga koristiš
- Ušteda: ostane u limitu od 500h! ✅

---

## 🔧 Advanced: Custom Domain (Optional)

Ako imaš svoj domain (npr. `mynews.com`):

1. Railway Settings → Networking
2. Klikni: **"Custom Domain"**
3. Dodaj: `mynews.com` ili `dashboard.mynews.com`
4. Postavi CNAME DNS record kako Railway kaže
5. Gotovo! Dashboard na tvom domenu! 🎉

---

## 🐛 Troubleshooting

### ❌ Problem: Build fails - "No module named 'torch'"
**Rešenje:** 
```bash
# Proveri requirements.txt
# Railway koristi Nixpacks koji automatski instalira sve
```

### ❌ Problem: Dashboard ne učitava podatke
**Rešenje:** 
```bash
# Proveri da li postoje fajlovi u data/processed/
# GitHub Actions treba da ih commit-uje!
# Možda trebaš ručno push-ovati postojeće podatke:

git add data/processed/*.csv
git add models/
git commit -m "Add initial data"
git push
```

### ❌ Problem: "This site can't be reached"
**Rešenje:** 
- Deployment traje 5-10 min pri prvom deploy-u
- Proveri Railway Logs za errore
- Verify da je PORT env variable = 8501

### ❌ Problem: Prelazi 500h/mesec
**Rešenje:** 
- Enable "Sleep after inactivity" u Settings
- Ili upgrade na Developer plan ($20/mesec)

---

## 🎯 Final Setup

Nakon što sve radi:

1. ✅ GitHub Actions → Update podataka svaki dan u 8:00 UTC
2. ✅ GitHub push → Commit novih podataka
3. ✅ Railway detektuje → Auto re-deploy
4. ✅ Dashboard live 24/7 → Fresh podaci svaki dan!

### URLs:
- **GitHub Repo:** `https://github.com/YOUR_USERNAME/news-trend-analysis`
- **Railway Dashboard:** `https://railway.app/dashboard`
- **Public App:** `https://news-trend-analysis-production.up.railway.app`

---

## 📈 Scaling (Future)

Kada projekat naraste:

**Opcija 1: Railway Developer Plan**
- $20/mesec
- Unlimited hours
- Better resources

**Opcija 2: Drugi hosting**
- Render.com (free tier 750h/mesec)
- Heroku (platforma, slično Railway-u)
- AWS EC2 / DigitalOcean (manual setup)

**Opcija 3: Docker + VPS**
- Hetzner VPS (~€5/mesec)
- Full control
- Docker container sa Streamlit

---

## 🎉 Gotovo!

Sada imaš:
- ✅ Automatski daily updates (GitHub Actions)
- ✅ 24/7 dashboard (Railway)
- ✅ Automatski deployment na svaki push
- ✅ Free hosting (sa limitima)

**Dashboard URL:**
```
https://news-trend-analysis-production.up.railway.app
```

Podeli sa svetom! 🌍✨
