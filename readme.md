# Gorodovoy Vision


Перед запуском в терминале ввести команды:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
Copy-Item .env.example .env
```

```powershell
python -m app
```

### Проверка состояния приложения:
http://127.0.0.1:8000/health

Для завершения приложения нажми Ctrl+C в терминале.