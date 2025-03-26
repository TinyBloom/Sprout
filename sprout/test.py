from sqlalchemy import create_engine, text

# Database connection settings
DB_NAME = "sprout_model"
DB_USER = "sprout_admin"
DB_PASSWORD = "sprout_pwd"
DB_HOST = "localhost"  # Use "my_postgres" if running inside Docker
DB_PORT = "5432"

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as conn:
        # result = conn.execute(text("SELECT NOW();"))
        result = conn.execute(text("SELECT * from model_files;"))
        print("Current Timestamp:", result.scalar())
except Exception as e:
    print("Error:", e)