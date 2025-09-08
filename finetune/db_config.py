import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', '117.50.163.224'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Ff12^&*3456fx'),
    'database': os.getenv('DB_NAME', 'zero')
}