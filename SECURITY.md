# Security Policy

## 🔒 Supported Versions

We currently support the following versions of this project with security updates:

| Version | Supported |
|---------|-----------|
| Latest (main) | ✅ |
| v1.x | ✅ |
| v0.x | ❌ |

## 🚨 Reporting a Vulnerability

We take the security of our project seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

### How to Report

1. **Do NOT** open a public issue
2. Email us at: **qqxmassimo@gmail.com**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Your contact information (optional)

### What to Expect

- We will acknowledge your report within 24 hours
- We will investigate and provide updates within 5 business days
- If confirmed, we will:
  - Assign a severity level
  - Work on a fix
  - Coordinate a responsible disclosure
  - Credit you in the release notes (if desired)

### Scope

This security policy applies to:

- ✅ Source code in this repository
- ✅ Build and release processes
- ✅ Dependencies (when we control the version)

This policy does NOT apply to:

- ❌ Third-party services or APIs
- ❌ User-provided data or configurations
- ❌ Social engineering attacks

## 🔐 Best Practices

To help keep this project secure:

1. **Keep dependencies updated**
   ```bash
   pip install -r requirements.txt
   ```

2. **Use environment variables for secrets**
   ```bash
   export API_KEY=your_secret_key
   ```

3. **Enable 2FA on your GitHub account**

4. **Review code before running locally**

## 📜 Disclosure Policy

We follow a coordinated disclosure process:

1. Vulnerability reported privately
2. We validate and investigate
3. We develop a fix
4. We notify the reporter
5. We release the fix with appropriate credit
6. We publish a security advisory

## 🙏 Thank You

We appreciate your help in keeping this project secure. Responsible security researchers will be acknowledged in our release notes.

---

*This security policy is adapted from the [GitHub Security Policy template](https://github.com/github/security-advisories/tree/master/.github/SECURITY.md).*
