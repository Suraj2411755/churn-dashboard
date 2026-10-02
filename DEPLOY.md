# Put the churn dashboard online (Streamlit Community Cloud - free)

1. Create a free account at https://github.com and https://share.streamlit.io (sign in with GitHub).
2. On GitHub: New repository -> name it `churn-dashboard` -> Public (or Private).
3. Upload EVERYTHING in this folder (drag & drop in the browser is fine), including the hidden
   `.streamlit` folder and `churn_model.joblib`. Do NOT upload the 79 MB raw CSV (GitHub limit is 100 MB
   and the app does not need it).
4. On share.streamlit.io: Create app -> pick the repository -> Main file path: `app.py`.
5. Open "Advanced settings" -> choose Python 3.12 -> Deploy. The first start takes a few minutes.
6. You get a link like https://your-name-churn-dashboard.streamlit.app - share that link.

Test after it opens: menu 3 (Predict churn) -> upload `demo_customers.csv`.

If the saved model does not load (version mismatch) the sidebar says "not trained yet": go to
menu 1 and train once by uploading your CSV.

Privacy: a public app can be opened by anyone. Never upload real customer phone numbers to it.
Use Private app sharing (Share button) if you need to restrict who can open it.
