# HTTPS Deployment

`03_web_app/web_app.py` now supports optional HTTPS through environment variables.

## Run with HTTPS directly

Set these environment variables before starting the app:

- `APP_HOST` - default `0.0.0.0`
- `APP_PORT` - default `5000`
- `SSL_CERTFILE` - path to your certificate file
- `SSL_KEYFILE` - path to your private key file

Example on Windows PowerShell:

```powershell
$env:APP_HOST = '0.0.0.0'
$env:APP_PORT = '5000'
$env:SSL_CERTFILE = 'C:\path\to\fullchain.pem'
$env:SSL_KEYFILE = 'C:\path\to\privkey.pem'
.\.venv310\Scripts\python.exe 03_web_app\web_app.py
```

Example on Linux:

```bash
export APP_HOST=0.0.0.0
export APP_PORT=5000
export SSL_CERTFILE=/etc/letsencrypt/live/your-domain/fullchain.pem
export SSL_KEYFILE=/etc/letsencrypt/live/your-domain/privkey.pem
python3 03_web_app/web_app.py
```

## Recommended cloud setup

If you are deploying to a public server, the usual production setup is:

1. Put Nginx in front of the Flask app.
2. Terminate TLS in Nginx with a real certificate from Let's Encrypt.
3. Proxy requests and WebSocket traffic to the Flask-SocketIO app on `127.0.0.1:5000`.

This is the most stable option for browser camera access because `getUserMedia()` requires HTTPS.

## Notes

- If you use a self-signed certificate, the browser will show a trust warning before camera access works.
- If you stay on plain HTTP, browser camera access will remain blocked on cloud hosts.
