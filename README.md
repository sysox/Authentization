# PoznejFI

Workshop notebooks for the FI MUNI security lab — identity verification and authentication.
See [ABSTRAKT.md](ABSTRAKT.md) for the Czech workshop description.

<p align="center">
  <img src="docs/qr.png" alt="QR code" width="220">
</p>

| Notebook | Topic |
|---|---|
| `Authentization.ipynb` | Passwords, OTP, certificates |
| `Biometrics.ipynb` | Fingerprint processing & fake-print attack reproduction |

---

## Run without installation

Launch the notebooks directly in your browser — no setup needed:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sysox/PoznejFI/HEAD)

> **Note:** Binder works for `PoznejFI.ipynb` (passwords, OTP, certificates).  
> It is **not suitable** for `biometrics.ipynb`, which requires uploading your own finger photos and local hardware access.

---

## Requirements

- Python **3.10+**
- Jupyter Lab or Jupyter Notebook

---

## Quick start

1. Clone the repo: `git clone https://github.com/sysox/PoznejFI.git`
2. Open a terminal — `Ctrl + T`
3. Type `./St`, press `Tab` to complete, then `Enter`

Jupyter Notebook opens in your browser automatically.

---

## Usage

1. Select kernel **Python (poznejfi)**.
2. Open a notebook and run cells top to bottom.

For `biometrics.ipynb`: put your finger photo(s) in the `data/` folder before starting.

The notebook reproduces the attack described in:
- BBC News — [Hacker 'fakes' German minister's fingerprints using photos](https://www.bbc.com/news/technology-30623611)
- Ars Technica — [Politician's fingerprint reproduced using photos of her hands](https://arstechnica.com/information-technology/2014/12/politicians-fingerprint-reproduced-using-photos-of-her-hands/)

---

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl + T` | Open a new terminal |
