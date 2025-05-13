from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

# Setup
DATABASE_URL = "sqlite:///component_images.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class TrainingImage(Base):
    __tablename__ = "training_images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)
    label = Column(String)



class ClassifiedImage(Base):
    __tablename__ = "classified_images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True)
    file_path = Column(String)
    predicted_class = Column(String)
    confidence = Column(Float)

# Create the table
Base.metadata.create_all(bind=engine)