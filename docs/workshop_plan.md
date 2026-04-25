# Workshop: Authentication and Keys 🔐

## 0. Password Database Leak
- **Task:** Show three toy databases:
  - database A: local usernames + plaintext passwords
  - database B: email logins + plaintext passwords
  - database C: local usernames + password-looking values that are not directly readable
- **Concept:** Plaintext leaks reveal passwords immediately. Email-based logins make it easy to try the same password on other services.
- **Takeaway:** We should not store passwords directly.

## 1. Am I compromised?
- **Task:** Check personal/school email on [haveibeenpwned.com](https://haveibeenpwned.com).
- **Concept:** Real breaches expose emails, password hashes, and sometimes plaintext passwords.
- **Takeaway:** Password reuse means one leaked password can compromise many accounts.

## 2. How to check a secret without sharing it?
- **Concept:** k-Anonymity.
- **Task:** Compute `SHA1(password)`, send only the first 5 hex characters, and check the returned suffixes locally.
- **Takeaway:** The full password never leaves the computer.

## 3. Password Hashes
- **Task:** Compute SHA-1 / SHA-256 hashes for a few passwords.
- **Task:** Change one character and observe how the hash changes.
- **Concept:** Better to store a one-way value than the password itself.
- **Takeaway:** Hashes hide the password, but leaked hashes can still be attacked.

## 4. Brute Force: The Offline Threat
- **Task:** Brute-force a short password and a SHA-1 hash.
- **Task:** Try a small dictionary attack against toy leaked hashes.
- **Takeaway:** If a hash database leaks, attackers can guess offline without contacting the original service.

## 5. Protected ZIP: Same Offline Problem
- **Task:** Open `file.zip`, try a wrong password, then extract it with password `abcd`.
- **Task:** Try a small dictionary attack against the ZIP password.
- **Takeaway:** Password-protected files are also vulnerable when the password is weak.

## 6. Bonus: Why Salt Exists
- **Task:** Show that the same password gives the same unsalted hash, but different salted hashes.
- **Takeaway:** Salt prevents equal-password leaks and precomputed tables, but weak passwords remain weak.

## 7. Dynamic Secrets: TOTP
- **Task:** Scan a QR code with a mobile Authenticator app.
- **Task:** Compare the 6-digit code on the phone with the Python output.
- **Under the hood:** Verify the math using manual HMAC-SHA1.
- **Takeaway:** Secrets that change every 30 seconds are harder to exploit than static passwords.

## 8. Asymmetric Proof: Public/Private Keys
- **Task:** Generate a keypair and sign a fresh server challenge.
- **Takeaway:** The secret never leaves your device; only the challenge and signature are transmitted.

## 9. Signatures and Certificates
- **Task:** Sign a document, verify it, then modify the document and verify failure.
- **Task:** Inspect a CA certificate and verify a user certificate.
- **Takeaway:** Signatures protect exact data; certificates bind public keys to identities.
