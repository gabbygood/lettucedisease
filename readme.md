#uvicorn lettucediseaseserver:app --host 192.168.145.43 --port 8000 --reload
#uvicorn plantclassifier:app --host 192.168.145.43 --port 8000 --reload

gunicorn -w 4 -k uvicorn.workers.UvicornWorker lettucediseaseserver:app


CLASS_NAMES = [
    'Fungal', 'healthy', 'Viral'
]
