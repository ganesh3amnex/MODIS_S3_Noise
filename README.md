# Crop Noise Explorer (Streamlit)

Taluk-level land-cover noise in crop grids, Kharif 2026-27.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a **private** GitHub repository and push this folder to it (app.py, data/, requirements.txt, .streamlit/).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, pick the repo, branch `main`, main file `app.py`.
3. After it deploys, open the app's **Settings → Sharing** to choose public (anyone with the link) or invite viewers by email.

All calculations run on the server; viewers receive only the rendered maps, metrics and table.
