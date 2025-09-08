import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'db-mysql.dian-stable.com'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'longquan'),
    'password': os.getenv('DB_PASSWORD', 'longquan'),
    'database': os.getenv('DB_NAME', 'test')
}