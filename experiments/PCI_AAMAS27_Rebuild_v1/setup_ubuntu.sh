#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
trap 'echo "SETUP FAILED at line $LINENO. No training was launched." >&2' ERR
if [ "$(uname -s)" != Linux ]; then
  echo "Use an Ubuntu 24.04 x86_64 IBM VM. This installer does not alter macOS." >&2; exit 2
fi
if [ "$(id -u)" -eq 0 ]; then SUDO=(); else SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y python3 python3-venv python3-pip build-essential pkg-config libssl-dev git curl unzip nodejs npm
python3 - <<'PY'
import sys
if not ((3,11)<=sys.version_info[:2]<=(3,12)):
    raise SystemExit('Use Python 3.11 or 3.12 for the pinned environment; Ubuntu 24.04 provides 3.12.')
PY
node -e 'if (+process.versions.node.split(".")[0]<18) throw Error("Node >=18 required")'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel setuptools
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
(
  cd crypto
  if [ -f package-lock.json ]; then npm ci; else npm install; fi
)
mkdir -p crypto/bin crypto/vendor
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl --proto '=https' --tlsv1.2 --fail --location https://sh.rustup.rs -o crypto/vendor/rustup-init.sh
  sh crypto/vendor/rustup-init.sh -y --profile minimal --default-toolchain 1.85.1 --no-modify-path
else
  "$HOME/.cargo/bin/rustup" toolchain install 1.85.1 --profile minimal
fi
if [ ! -x crypto/bin/circom ]; then
  if [ ! -d crypto/vendor/circom/.git ]; then
    git clone --depth 1 --branch v2.2.2 https://github.com/iden3/circom.git crypto/vendor/circom
  fi
  (cd crypto/vendor/circom && "$HOME/.cargo/bin/cargo" +1.85.1 build --release --locked)
  cp crypto/vendor/circom/target/release/circom crypto/bin/circom
fi
crypto/bin/circom --version
node -e 'console.log("snarkjs",JSON.parse(require("fs").readFileSync("crypto/node_modules/snarkjs/package.json","utf8")).version)' 
.venv/bin/python -m pip freeze > requirements-lock.txt
mkdir -p validation
{
  date -u
  node --version
  npm --version
  "$HOME/.cargo/bin/rustc" +1.85.1 --version
  crypto/bin/circom --version
  git -C crypto/vendor/circom rev-parse HEAD 2>/dev/null || true
} > validation/server_toolchain.txt
.venv/bin/python -m pytest -q tests
printf '\nSETUP COMPLETE. Next: bash run_all.sh --profile smoke --jobs 2 --results results/smoke\n'
