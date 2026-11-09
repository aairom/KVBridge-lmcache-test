#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/github-push.sh — Universal script to push a project to GitHub
#
# Usage: ./scripts/github-push.sh
#   Run from anywhere inside the project; it always operates on the project root.
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ── 1. Initialise if .git doesn't exist ──────────────────────────────────────
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Initialising new Git repository…"
    git init
    git branch -M main
else
    echo "Existing Git repository detected."
fi

# ── 2. Configure remote ───────────────────────────────────────────────────────
CURRENT_REMOTE="$(git remote get-url origin 2>/dev/null || true)"

if [ -z "$CURRENT_REMOTE" ]; then
    # No remote set — ask for URL
    while true; do
        read -rp "Enter the GitHub repository URL: " repo_url
        if [ -n "$repo_url" ]; then break; fi
        echo "⚠️  URL cannot be empty. Please enter a valid GitHub URL."
    done
    git remote add origin "$repo_url"
    echo "Remote 'origin' added: $repo_url"
else
    echo "Remote 'origin' is: $CURRENT_REMOTE"
    read -rp "Change remote URL? Press Enter to keep, or type a new URL: " new_url
    if [ -n "$new_url" ]; then
        git remote set-url origin "$new_url"
        echo "Remote updated to: $new_url"
    fi
fi

# ── 3. Stage all changes ──────────────────────────────────────────────────────
git add .

# ── 4. Commit only if there is something to commit ───────────────────────────
if git diff --cached --quiet; then
    echo "ℹ️  Nothing new to commit — working tree is clean."
else
    read -rp "Commit message (default: 'update'): " commit_msg
    commit_msg="${commit_msg:-update}"
    git commit -m "$commit_msg"
fi

# ── 5. Push to current branch ─────────────────────────────────────────────────
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Pushing branch '$BRANCH' to origin…"
git push -u origin "$BRANCH"

echo ""
echo "✅  Done! View your repository on GitHub."
