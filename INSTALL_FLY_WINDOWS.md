# Installing Fly CLI on Windows

Your antivirus blocked the automatic installer. Here are alternative methods:

## Method 1: Download Manual Installer (Easiest)

1. **Go to**: https://github.com/superfly/flyctl/releases/latest
2. **Download**: `flyctl_x.x.x_windows_amd64.zip` (or the latest version)
3. **Extract** the zip file
4. **Copy** `flyctl.exe` to a folder in your PATH, such as:
   - `C:\Windows\System32\` (requires admin)
   - Or create `C:\flyctl\` and add it to PATH

5. **Add to PATH** (if using custom folder):
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"
   - Under "User variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\flyctl\`
   - Click OK on all dialogs

6. **Verify**:
   ```powershell
   fly version
   ```

## Method 2: Using Chocolatey (if you have it)

```powershell
choco install flyctl
```

## Method 3: Using Scoop (if you have it)

```powershell
scoop install flyctl
```

## Method 4: Allow PowerShell Script (if you trust it)

1. Open PowerShell as Administrator
2. Run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Then try the install again:
   ```powershell
   iwr https://fly.io/install.ps1 -useb | iex
   ```

---

**After installation, verify:**
```powershell
fly version
```

Then continue with the deployment steps!

