# GitHub Actions Setup Guide

## ✅ What Was Fixed

The GitHub Actions workflow file (`.github/workflows/daily-update.yml`) had YAML indentation errors. These have been fixed:

- ✅ Fixed indentation for `on:`, `jobs:`, and `steps:` sections
- ✅ Changed from 4-space to 2-space indentation (YAML standard)
- ✅ Added environment variable for NEWS_API_KEY
- ✅ All YAML syntax errors resolved

## 🔧 Required Setup in GitHub

To use the GitHub Actions workflow, you need to configure secrets in your GitHub repository:

### 1. Add NEWS_API_KEY Secret

1. Go to your GitHub repository
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add:
   - **Name**: `NEWS_API_KEY`
   - **Value**: Your NewsAPI key from https://newsapi.org/

### 2. Enable GitHub Actions

1. Go to the **Actions** tab in your repository
2. If actions are disabled, click **Enable Actions**
3. The workflow will appear in the list

## 📅 Workflow Schedule

The workflow is configured to:
- **Automatic**: Run daily at 8:00 AM UTC (configured via cron)
- **Manual**: Can be triggered manually from the Actions tab

## 🚀 Manual Trigger

To run the workflow manually:
1. Go to **Actions** tab
2. Select "Daily DVC Pipeline Update" workflow
3. Click **Run workflow** button
4. Select branch (usually `main`)
5. Click **Run workflow**

## 📝 What the Workflow Does

1. **Checkout repository** - Gets the latest code
2. **Set up Python 3.10** - Installs Python environment
3. **Install dependencies** - Installs DVC and requirements
4. **Run DVC pipeline** - Executes the data pipeline:
   - Fetches new articles
   - Preprocesses data
   - Runs sentiment analysis
   - Generates summaries
   - Trains topic model
5. **Commit changes** - Automatically commits and pushes results

## ⚙️ Workflow Configuration

```yaml
# Run daily at 8:00 AM UTC
on:
  schedule:
    - cron: '0 8 * * *'
  workflow_dispatch:  # Manual trigger
```

To change the schedule, edit the cron expression:
- `'0 8 * * *'` = 8:00 AM UTC daily
- `'0 */6 * * *'` = Every 6 hours
- `'0 0 * * 0'` = Weekly on Sunday at midnight

## 🔍 Viewing Results

After the workflow runs:
1. Go to **Actions** tab
2. Click on the workflow run
3. View logs for each step
4. Check committed changes in your repository

## 🛠️ Troubleshooting

### Workflow fails with "API key error"
- Check that `NEWS_API_KEY` secret is set correctly
- Verify your API key is valid at https://newsapi.org/

### Workflow fails at "dvc pull"
- First run may fail if DVC remote is not configured
- You may need to set up DVC remote storage (S3, GCS, etc.)
- Or remove the `dvc pull` line if not using remote storage

### Workflow fails at "git push"
- Ensure GitHub Actions has write permissions
- Go to **Settings** → **Actions** → **General**
- Under "Workflow permissions", select "Read and write permissions"

## 📊 DVC Remote (Optional)

If you want to use DVC with remote storage:

```bash
# Configure DVC remote (example with S3)
dvc remote add -d myremote s3://mybucket/path

# Or with Google Drive
dvc remote add -d myremote gdrive://your-folder-id

# Push data to remote
dvc push
```

Then update `.github/workflows/daily-update.yml` to include authentication for your remote storage.

## ✅ Verification

To verify the workflow is set up correctly:
1. Check that the YAML file has no syntax errors (✅ Already fixed)
2. Ensure secrets are configured in GitHub
3. Run the workflow manually first
4. Check the logs for any errors
5. Verify new data is committed to the repository

---

**Status**: ✅ Workflow file is fixed and ready to use!

Remember to add your `NEWS_API_KEY` secret in GitHub repository settings.
