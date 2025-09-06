import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    UPLOAD_FOLDER = 'uploads/'
    ALLOWED_EXTENSIONS = {'py'}
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')