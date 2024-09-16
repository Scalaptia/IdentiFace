# API detección de caras y sexo

## Uso

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar servidor

```bash
uvicorn app:app --reload
```

## Endpoints

### POST /predict

File:

```bash
curl -X POST "http://localhost:8000/predict/" -F "file=@path/to/image"
```

URL:

```bash
curl -X POST "http://localhost:8000/predict/" -d "url=https://example.com/image.jpg"
```
