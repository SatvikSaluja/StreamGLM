#!/usr/bin/env bash
# Publish validated static evidence to the existing gh-pages branch.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_pages.py
remote=$(git remote get-url origin)
publish_dir=$(mktemp -d)
# Keep the checkout available if pushing fails, for inspection or retry.
git clone --depth 1 --branch gh-pages "$remote" "$publish_dir"
cp -a _site/. "$publish_dir/"
git -C "$publish_dir" add .
if git -C "$publish_dir" diff --cached --quiet; then
  echo "Published content is already current."
  exit 0
fi
git -C "$publish_dir" commit -m "Publish validated results from $(git rev-parse --short HEAD)"
git -C "$publish_dir" push origin gh-pages
echo "Published; GitHub Pages will build the gh-pages branch. Checkout: $publish_dir"
