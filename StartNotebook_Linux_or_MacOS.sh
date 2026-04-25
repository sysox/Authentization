#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PoznejFI - unified setup + launch script
# ============================================================
# Features:
# - create / copy / reuse virtual environment
# - install requirements
# - register Jupyter kernel
# - launch Jupyter Notebook
#
# Example usage:
#   ./StartNotebook_Linux_or_MacOS.sh
#   ./StartNotebook_Linux_or_MacOS.sh --mode standard
#   ./StartNotebook_Linux_or_MacOS.sh --mode copy --copy-from /path/to/prebuilt/env
#   ./StartNotebook_Linux_or_MacOS.sh --no-jupyter
# ============================================================

# ----------------------------
# Configuration & Defaults
# ----------------------------
MODE="auto"
VENV_DIR=".venv"
REQ_FILE="requirements.txt"
COPY_FROM=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUPYTER_CMD="jupyter notebook --ServerApp.token='' --ServerApp.password=''"
START_JUPYTER=1
REGISTER_KERNEL=1
KERNEL_NAME="poznejfi"

DATA_DIR="$PROJECT_ROOT/data"

# Helper functions
die()   { echo "[ERROR] $*" >&2; exit 1; }
info()  { echo "[INFO]  $*"; }
debug() { echo "[DEBUG] $*"; }

# ----------------------------
# Argument Parsing
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="${2:-}";        shift 2 ;;
    --venv|--venv-dir) VENV_DIR="${2:-}"; shift 2 ;;
    --requirements|--req) REQ_FILE="${2:-}"; shift 2 ;;
    --copy-from)   COPY_FROM="${2:-}";   shift 2 ;;
    --jupyter)     JUPYTER_CMD="${2:-}"; shift 2 ;;
    --no-jupyter)  START_JUPYTER=0;      shift 1 ;;
    --no-kernel)   REGISTER_KERNEL=0;   shift 1 ;;
    -h|--help)
      cat <<EOF
Usage:
  $0 [options]

Options:
  --mode auto|standard|copy   Virtualenv mode (default: auto)
  --venv-dir PATH             Target venv directory (default: .venv)
  --requirements PATH         Requirements file (default: requirements.txt)
  --copy-from PATH            Copy prebuilt venv from PATH (used with --mode copy)
  --jupyter CMD               Custom Jupyter launch command
  --no-jupyter                Run setup only, do not launch Jupyter
  --no-kernel                 Skip ipykernel registration
  -h, --help                  Show this help
EOF
      exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# Normalize paths relative to project root
[[ "$VENV_DIR" == /* ]] || VENV_DIR="$PROJECT_ROOT/$VENV_DIR"
[[ "$REQ_FILE" == /* ]] || REQ_FILE="$PROJECT_ROOT/$REQ_FILE"

# ----------------------------
# Functions
# ----------------------------

venv_exists() {
  [[ -f "${VENV_DIR}/bin/activate" ]]
}

copy_venv() {
  [[ -n "$COPY_FROM" ]] || die "--mode copy requires --copy-from <path>"
  [[ -d "$COPY_FROM" ]]  || die "Copy source not found: $COPY_FROM"
  info "Copying venv: $COPY_FROM -> $VENV_DIR"
  rm -rf "$VENV_DIR"
  cp -a "$COPY_FROM" "$VENV_DIR"
  chmod +x "${VENV_DIR}/bin/"* 2>/dev/null || true
}

create_standard_venv() {
  info "Creating virtual environment: $VENV_DIR"
  python3 -m venv "$VENV_DIR" || die "Failed. Is python3-venv installed?"
}

activate_venv() {
  debug "Activating: ${VENV_DIR}/bin/activate"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
}

install_python_deps() {
  info "Upgrading pip and wheel..."
  python -m pip install --upgrade pip wheel -q

  if [[ -f "$REQ_FILE" ]]; then
    info "Installing requirements from $REQ_FILE..."
    python -m pip install -r "$REQ_FILE"
  else
    info "No requirements file found — skipping."
  fi

  info "Ensuring Jupyter packages..."
  python -m pip install notebook ipykernel -q
}

register_kernel() {
  if [[ "$REGISTER_KERNEL" -eq 1 ]]; then
    info "Registering Jupyter kernel: $KERNEL_NAME"
    python -m ipykernel install --user \
      --name "$KERNEL_NAME" \
      --display-name "Python ($KERNEL_NAME)" || true
  else
    info "Skipping kernel registration."
  fi
}

create_data_dirs() {
  if [[ ! -d "$DATA_DIR" ]]; then
    mkdir -p "$DATA_DIR"
    info "Created: $DATA_DIR"
  else
    info "data/ already exists — nothing to create."
  fi
}

print_summary() {
  cat <<EOF

============================================================
Setup complete
============================================================

Project root:
  $PROJECT_ROOT

Virtual environment:
  $VENV_DIR

Data folder:
  $DATA_DIR/
    Place your finger photos here before opening biometrics.ipynb.

Notebooks:
  PoznejFI.ipynb      Passwords, OTP, certificates, KeePass
  biometrics.ipynb    Fingerprint processing & fake-print DIY

Manual launch later:
  source "$VENV_DIR/bin/activate"
  cd "$PROJECT_ROOT"
  jupyter notebook

Inside Jupyter select kernel:
  Python ($KERNEL_NAME)

EOF
}

launch_jupyter() {
  if [[ "$START_JUPYTER" -eq 1 ]]; then
    info "Changing to project root: $PROJECT_ROOT"
    cd "$PROJECT_ROOT"
    info "Launching: $JUPYTER_CMD"
    exec $JUPYTER_CMD
  else
    info "Setup finished — Jupyter not started (--no-jupyter)."
  fi
}

# ----------------------------
# Main
# ----------------------------

debug "Project root : $PROJECT_ROOT"
debug "Mode         : $MODE"

case "$MODE" in
  copy)
    copy_venv ;;
  standard)
    create_standard_venv ;;
  auto)
    if venv_exists; then
      info "Existing venv found: $VENV_DIR"
    elif [[ -n "$COPY_FROM" && -d "$COPY_FROM" ]]; then
      info "No venv found, copy source available — copying..."
      copy_venv
    else
      info "No venv found — creating standard venv..."
      create_standard_venv
    fi ;;
  *)
    die "Invalid --mode: $MODE (use auto|standard|copy)" ;;
esac

activate_venv
install_python_deps
register_kernel
create_data_dirs
print_summary
launch_jupyter
