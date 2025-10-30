# 🤖 Automatizacija - Daily News Updates

Postoje **3 načina** da automatizuješ daily updates:

---

## ⏰ Opcija 1: Cron Job (Lokalno - Najjednostavnije)

Pokreće se **na tvom računaru** svaki dan u određeno vreme.

### Setup:
```bash
chmod +x setup_cron.sh
./setup_cron.sh
```

### Konfigurisanje vremena:
Otvori crontab:
```bash
crontab -e
```

Promeni liniju (trenutno 8:00 AM):
```bash
# Min Hour Day Month Weekday Command
0 8 * * * cd /home/David/Documents/GithubRepos/news-trend-analysis && .venv/bin/python daily_update.py --max-results 50 >> logs/cron_update.log 2>&1
```

**Primeri:**
- `0 8 * * *` = 8:00 AM svaki dan
- `0 */6 * * *` = Svaka 6 sati
- `0 8,20 * * *` = 8:00 AM i 8:00 PM
- `0 8 * * 1-5` = 8:00 AM radnim danima

### Provera:
```bash
crontab -l  # Lista svih cron jobova
tail -f logs/cron_update.log  # Live log
```

### Uklanjanje:
```bash
crontab -e  # Delete the line
```

**✅ Prednosti:** Jednostavno, brzo  
**❌ Nedostaci:** Računar mora biti uključen

---

## 🐧 Opcija 2: Systemd Timer (Lokalno - Najnapredniji)

Bolje od cron - automatski restart ako padne, bolji logging.

### Setup:
```bash
chmod +x setup_systemd.sh
./setup_systemd.sh
```

### Komande:
```bash
# Status
systemctl --user status news-update.timer

# Logovi
journalctl --user -u news-update.service -f

# Stop
systemctl --user stop news-update.timer

# Start
systemctl --user start news-update.timer

# Disable
systemctl --user disable news-update.timer

# Ručno pokretanje
systemctl --user start news-update.service
```

### Konfigurisanje vremena:
```bash
nano ~/.config/systemd/user/news-update.timer
```

Promeni `OnCalendar`:
```ini
[Timer]
OnCalendar=daily           # Svaki dan u ponoć
OnCalendar=08:00          # Svaki dan u 8:00
OnCalendar=*-*-* 08:00    # Isto kao gore
OnCalendar=Mon 08:00      # Samo ponedeljkom u 8:00
OnCalendar=*-*-* 8,20:00  # Svaki dan u 8:00 i 20:00
Persistent=true           # Pokreni ako je propušteno
```

Nakon izmene:
```bash
systemctl --user daemon-reload
systemctl --user restart news-update.timer
```

**✅ Prednosti:** Naprednije, bolji logging, automatski restart  
**❌ Nedostaci:** Računar mora biti uključen

---

## ☁️ Opcija 3: GitHub Actions (Cloud - Besplatno)

Pokreće se **na GitHub serverima** - ne treba ti računar!

### Setup:

1. **Push kod na GitHub:**
   ```bash
   git add .
   git commit -m "Add automatic updates"
   git push
   ```

2. **Dodaj API key kao GitHub Secret:**
   - Idi na: `https://github.com/YOUR_USERNAME/news-trend-analysis/settings/secrets/actions`
   - Klikni: **New repository secret**
   - Name: `NEWS_API_KEY`
   - Value: `YOUR_NEWSDATA_IO_API_KEY` (get from https://newsdata.io)
   - Klikni: **Add secret**

3. **Gotovo!** 🎉

### Kako radi:
- ✅ Pokreće se **svaki dan u 8:00 UTC** (9:00 CET, 10:00 CEST)
- ✅ GitHub Actions automatski fetch-uje nove članke
- ✅ Automatski commit-uje i push-uje promene
- ✅ Besplatno za javne repo-e (2000 min/mesec za privatne)

### Ručno pokretanje:
1. Idi na: `https://github.com/YOUR_USERNAME/news-trend-analysis/actions`
2. Klikni: **Daily News Update** workflow
3. Klikni: **Run workflow** → **Run workflow**

### Konfigurisanje vremena:
Otvori `.github/workflows/daily-update.yml`:
```yaml
on:
  schedule:
    - cron: '0 8 * * *'  # 8:00 UTC svaki dan
```

**Primeri:**
- `0 */6 * * *` = Svaka 6 sati
- `0 8,20 * * *` = 8:00 i 20:00 UTC
- `0 8 * * 1-5` = Radnim danima u 8:00 UTC

### Logovi:
GitHub Actions tab → Daily News Update → Najnoviji run

**✅ Prednosti:** Radi 24/7, ne treba ti računar, besplatno  
**❌ Nedostaci:** Zahteva GitHub repo, ograničeno besplatnih minuta

---

## 🎯 Preporuka

**Za development (testiranje):**  
→ **Cron Job** - brzo setup, lako testiranje

**Za produkciju (lokalno):**  
→ **Systemd Timer** - napredniji, bolji logging

**Za produkciju (cloud):**  
→ **GitHub Actions** - radi 24/7 bez da ti računar bude uključen

---

## 📊 Testiranje

Ručno pokretanje (za test):
```bash
# Direktno
python daily_update.py --max-results 10

# Sa cron-om
bash -c "cd /home/David/Documents/GithubRepos/news-trend-analysis && .venv/bin/python daily_update.py --max-results 10"

# Sa systemd-om
systemctl --user start news-update.service
```

Provera logova:
```bash
# Cron
tail -f logs/cron_update.log

# Systemd
journalctl --user -u news-update.service -f

# GitHub Actions
# GitHub → Actions tab → Workflow run
```

---

## 🚀 Dashboard Always-On

Ako želiš da **dashboard radi 24/7**, imaš opcije:

1. **Lokalno sa systemd:**
   ```bash
   # Kreirati news-dashboard.service koji pokreće Streamlit
   systemctl --user enable news-dashboard.service
   ```

2. **Railway Deployment:**
   - Push na GitHub
   - Connectuj na Railway
   - Automatski deploy
   - Free tier: 500h/mesec

3. **Docker + VPS:**
   - Kontejnerizovati app
   - Deploy na jeftini VPS (Hetzner, DigitalOcean)

Hoćeš li da setupujem nešto od ovoga? 🤔
