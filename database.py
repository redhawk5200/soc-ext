from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from contextlib import contextmanager

# Database connection setup
DATABASE_URL = "postgresql+psycopg2://myuser:mypassword@localhost/mydatabase"
engine = create_engine(DATABASE_URL, echo=True)

# Create session factory
SessionLocal = sessionmaker(bind=engine)

# ✅ Create a global session instance that can be reused
session = SessionLocal()

@contextmanager
def get_db_session():
    """
    Context manager that yields a shared database session and ensures it is properly managed.
    """
    try:
        yield session  # ✅ Provide the global session
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Database operation failed: {e}")
    finally:
        pass  # ❌ Do NOT close the session, since it's meant to be reused


# def insert_dummy_data():
#     """
#     Inserts dummy ControlObjective and CUEC records with document metadata.
#     """
#     dummy_data = [
#         ControlObjective(
#             kb_doc_id="KB-001",
#             number=1,
#             description="Audit Compliance Check",
#             supporting_control_activities=["Activity 1", "Activity 2"],
#             document_metadata={
#                 "Document Title": "Financial Audit Report 2024",
#                 "Time Period": "Q1 2024",
#                 "Auditor Name": "John Doe",
#                 "Company Name": "XYZ Corporation",
#                 "Page Number": "15"
#             },
#             created_by="user_1",
#             created_at=datetime.utcnow(),
#             updated_at=datetime.utcnow()
#         ),
#         ControlObjective(
#             kb_doc_id="KB-002",
#             number=2,
#             description="Security Risk Assessment",
#             supporting_control_activities=["Activity 1", "Activity 2"],
#             document_metadata={
#                 "Document Title": "Cybersecurity Risk Analysis",
#                 "Time Period": "2023",
#                 "Auditor Name": "Jane Smith",
#                 "Company Name": "ABC Ltd",
#                 "Page Number": "25"
#             },
#             created_by="user_2",
#             created_at=datetime.utcnow(),
#             updated_at=datetime.utcnow()
#         ),
#     ]

#     with get_db_session() as session:
#         try:
#             session.add_all(dummy_data)
#             session.commit()
#             print("✅ Dummy data inserted successfully!")
#         except Exception as e:
#             session.rollback()
#             print(f"❌ Failed to insert data: {e}")
#         finally:
#             session.close()  # ✅ Close session to avoid memory leaks



def create_tables():
    """Creates all tables explicitly."""
    SQLModel.metadata.create_all(engine)  # Auto-detects all imported SQLModel tables
    print("✅ Tables created successfully!")

if __name__ == "__main__":
    create_tables()  # Create tables when running this script
    print("✅ Database setup completed!")

