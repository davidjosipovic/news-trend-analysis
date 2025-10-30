#!/bin/bash
# Setup cron job for daily news updates

PROJECT_DIR="/home/David/Documents/GithubRepos/news-trend-analysis"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
SCRIPT="$PROJECT_DIR/daily_update.py"
LOG_FILE="$PROJECT_DIR/logs/cron_update.log"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Cron job command (runs daily at 8:00 AM)
CRON_CMD="0 8 * * * cd $PROJECT_DIR && $VENV_PYTHON $SCRIPT --max-results 50 >> $LOG_FILE 2>&1"

# Add to crontab
(crontab -l 2>/dev/null | grep -v "$SCRIPT"; echo "$CRON_CMD") | crontab -

echo "✅ Cron job setup complete!"
echo "   Schedule: Daily at 8:00 AM"
echo "   Command: $VENV_PYTHON $SCRIPT --max-results 50"
echo "   Logs: $LOG_FILE"
echo ""
echo "To view cron jobs: crontab -l"
echo "To edit cron jobs: crontab -e"
echo "To remove this job: crontab -e (then delete the line)"
