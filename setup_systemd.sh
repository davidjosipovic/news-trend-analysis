#!/bin/bash
# Setup systemd timer for daily news updates

PROJECT_DIR="/home/David/Documents/GithubRepos/news-trend-analysis"

echo "Setting up systemd timer for automatic daily updates..."

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Copy service files to systemd user directory
mkdir -p ~/.config/systemd/user
cp "$PROJECT_DIR/news-update.service" ~/.config/systemd/user/
cp "$PROJECT_DIR/news-update.timer" ~/.config/systemd/user/

# Reload systemd daemon
systemctl --user daemon-reload

# Enable and start the timer
systemctl --user enable news-update.timer
systemctl --user start news-update.timer

echo ""
echo "✅ Systemd timer setup complete!"
echo "   Schedule: Daily at 8:00 AM"
echo "   Logs: $PROJECT_DIR/logs/systemd_update.log"
echo ""
echo "Useful commands:"
echo "  Check status:  systemctl --user status news-update.timer"
echo "  View logs:     journalctl --user -u news-update.service"
echo "  Stop timer:    systemctl --user stop news-update.timer"
echo "  Disable timer: systemctl --user disable news-update.timer"
echo "  Run manually:  systemctl --user start news-update.service"
echo ""
echo "Timer info:"
systemctl --user list-timers news-update.timer
