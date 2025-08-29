import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', '8.216.81.x'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Ff123456fx'),
    'database': os.getenv('DB_NAME', 'zero')
}