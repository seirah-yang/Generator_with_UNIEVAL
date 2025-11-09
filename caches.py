#!/bin/bash
# ==============================================================
# 🧹 clean_safe_caches.sh — macOS 안전 캐시 정리 스크립트
# 만든이: 양소라님용 / 목적: 캐시만 안전하게 삭제
# ==============================================================

echo "🧹 macOS 안전 캐시 정리 시작..."

# 관리자 권한 요청
sudo -v

# ------------------------------
# 1️⃣ 사용자 캐시 정리
# ------------------------------
echo "📂 사용자 캐시 정리 중..."
rm -rf ~/Library/Caches/* >/dev/null 2>&1
rm -rf ~/Library/Application\ Support/Code/Cache* >/dev/null 2>&1
rm -rf ~/Library/Application\ Support/Caput >/dev/null 2>&1
rm -rf ~/Library/Application\ Support/com.microsoft.VSCode* >/dev/null 2>&1

# ------------------------------
# 2️⃣ 시스템 캐시 중 안전 영역 정리
# ------------------------------
echo "🧩 시스템 캐시 정리 중..."
sudo rm -f /Library/Caches/Folders.h \
            /Library/Caches/macos.py \
            /Library/Caches/NSPersistentStoreCoordinator.h >/dev/null 2>&1

# ------------------------------
# 3️⃣ 브라우저 캐시 (Safari, Chrome, Edge)
# ------------------------------
echo "🌐 브라우저 캐시 정리 중..."
rm -rf ~/Library/Caches/com.apple.Safari >/dev/null 2>&1
rm -rf ~/Library/Application\ Support/Google/Chrome/Default/Cache >/dev/null 2>&1
rm -rf ~/Library/Application\ Support/Microsoft\ Edge/Default/Cache >/dev/null 2>&1

# ------------------------------
# 4️⃣ 로그 및 이전 업데이트 잔여파일 정리
# ------------------------------
echo "🪵 로그 캐시 정리 중..."
rm -rf ~/Library/Logs/*
sudo rm -rf /private/var/log/* >/dev/null 2>&1

# ------------------------------
# 5️⃣ 휴지통 비우기
# ------------------------------
echo "🗑️ 휴지통 비우는 중..."
sudo rm -rf ~/.Trash/* >/dev/null 2>&1

echo "✅ 모든 캐시 및 로그 정리 완료!"
echo "💡 권장: Mac을 재시작하면 시스템 캐시가 자동으로 재구성됩니다."