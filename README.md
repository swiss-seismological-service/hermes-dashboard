# hermes-dashboard

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 2. Configure the environment file

```bash
cp .env.example .env
```
Replace the values in the `.env` file with your own.

### 3. Run the application

```bash
python3 -m streamlit run app/main.py
```